import json
import re
import boto3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from langchain_aws import BedrockEmbeddings, ChatBedrock


# Configuration
BUCKET_NAME = 'medlaunch-rag'
EMBEDDINGS_INDEX_KEY = 'embeddings/embeddings_index.json'
EMBEDDING_MODEL = 'amazon.titan-embed-text-v1'
LLM_MODEL = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'  # Inference profile ARN
TOP_K_RESULTS = 5


def get_s3_client():
    """Initialize S3 client."""
    return boto3.client('s3')


def get_bedrock_embeddings():
    """Initialize Bedrock embeddings."""
    return BedrockEmbeddings(
        model_id=EMBEDDING_MODEL,
        region_name='us-east-1'
    )


def get_bedrock_llm():
    """Initialize Bedrock LLM (Claude)."""
    return ChatBedrock(
        model_id=LLM_MODEL,
        region_name='us-east-1',
        model_kwargs={
            "temperature": 0.1,
            "max_tokens": 2000
        }
    )


def load_embeddings_index(bucket: str, key: str) -> Dict:
    """
    Load embeddings index from S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 key for embeddings index
        
    Returns:
        Embeddings index dictionary
    """
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    index_data = json.loads(response['Body'].read().decode('utf-8'))
    return index_data


def detect_query_type(query: str) -> Tuple[str, Optional[str]]:
    """
    Detect if query is Q&A or Citation mode.
    
    Args:
        query: User query
        
    Returns:
        Tuple of (query_type, chapter_id)
        - query_type: "citation" or "question"
        - chapter_id: Extracted chapter ID if citation mode, None otherwise
    """
    query_lower = query.lower()
    
    # Citation mode patterns
    citation_patterns = [
        r'show me (?:chapter|standard|section)?\s*([A-Z]{2,3}[.-]\d+)',
        r'(?:cite|get|retrieve|display)\s*(?:chapter|standard|section)?\s*([A-Z]{2,3}[.-]\d+)',
        r'what does (?:chapter|standard|section)?\s*([A-Z]{2,3}[.-]\d+)\s*say',
        r'give me (?:the )?exact text (?:for|of|from)?\s*(?:chapter|standard|section)?\s*([A-Z]{2,3}[.-]\d+)',
        r'(?:chapter|standard|section)\s*([A-Z]{2,3}[.-]\d+)',
        r'verbatim (?:text|language|content) (?:for|of|from)?\s*([A-Z]{2,3}[.-]\d+)'
    ]
    
    for pattern in citation_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            chapter_id = match.group(1).upper()
            # Normalize chapter ID (replace - with .)
            chapter_id = chapter_id.replace('-', '.')
            return ("citation", chapter_id)
    
    return ("question", None)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def search_by_chapter(chunks: List[Dict], chapter_id: str) -> Optional[Dict]:
    """
    Search for a specific chapter by ID (exact match).
    
    Args:
        chunks: List of all chunks
        chapter_id: Chapter identifier to find
        
    Returns:
        Matching chunk or None
    """
    for chunk in chunks:
        if chunk['metadata']['chapter'] == chapter_id:
            return chunk
    return None


def vector_search(query_embedding: List[float], chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Perform vector similarity search.
    
    Args:
        query_embedding: Query embedding vector
        chunks: List of all chunks with embeddings
        top_k: Number of top results to return
        
    Returns:
        List of top K chunks with similarity scores
    """
    results = []
    
    for chunk in chunks:
        if chunk.get('embedding'):
            similarity = cosine_similarity(query_embedding, chunk['embedding'])
            results.append({
                'chunk': chunk,
                'similarity': similarity
            })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results[:top_k]


def handle_citation_mode(chunks: List[Dict], chapter_id: str, query: str) -> Dict:
    """
    Handle citation mode query - return exact text.
    
    Args:
        chunks: List of all chunks
        chapter_id: Chapter identifier to retrieve
        query: Original query
        
    Returns:
        Citation response dictionary
    """
    # Search for exact chapter
    chunk = search_by_chapter(chunks, chapter_id)
    
    if not chunk:
        return {
            "query": query,
            "query_type": "citation",
            "chapter": chapter_id,
            "found": False,
            "error": f"Chapter {chapter_id} not found in the document",
            "suggestion": "Please verify the chapter ID and try again"
        }
    
    # Return exact text without AI modification
    return {
        "query": query,
        "query_type": "citation",
        "chapter": chapter_id,
        "exact_text": chunk['text'],
        "source": {
            "document": chunk['metadata']['document'],
            "section": chunk['metadata']['section'],
            "chapter": chunk['metadata']['chapter'],
            "chunk_id": chunk['chunk_id']
        },
        "disclaimer": f"Exact text from NIAHO standards document - retrieved {datetime.now().isoformat()}"
    }


def handle_question_mode(query: str, chunks: List[Dict], embeddings_model: BedrockEmbeddings, llm: ChatBedrock) -> Dict:
    """
    Handle Q&A mode query - use RAG to generate answer.
    
    Args:
        query: User question
        chunks: List of all chunks
        embeddings_model: Embeddings model
        llm: Language model
        
    Returns:
        Q&A response dictionary
    """
    # Generate query embedding
    query_embedding = embeddings_model.embed_query(query)
    
    # Retrieve top K relevant chunks
    search_results = vector_search(query_embedding, chunks, top_k=TOP_K_RESULTS)
    
    # Build context from retrieved chunks
    context_parts = []
    citations = []
    
    for i, result in enumerate(search_results):
        chunk = result['chunk']
        similarity = result['similarity']
        
        context_parts.append(f"[Source {i+1} - Chapter {chunk['metadata']['chapter']}]\n{chunk['text']}\n")
        
        citations.append({
            "chunk_id": chunk['chunk_id'],
            "document": chunk['metadata']['document'],
            "section": chunk['metadata']['section'],
            "chapter": chunk['metadata']['chapter'],
            "relevance_score": round(similarity, 3)
        })
    
    context = "\n".join(context_parts)
    
    # Create RAG prompt
    prompt = f"""You are a healthcare compliance assistant. Answer the question based on the provided context from NIAHO (National Integrated Accreditation for Healthcare Organizations) standards.

        Context from NIAHO Standards:
        {context}

        Question: {query}

        Instructions:
        1. Answer the question based ONLY on the context provided above
        2. Cite specific chapters (e.g., "According to QM.1..." or "As stated in MM.4...")
        3. If the context doesn't contain enough information, say so clearly
        4. Be concise but comprehensive
        5. Use professional healthcare compliance language

        Answer:"""

    # Generate response
    response = llm.invoke(prompt)
    answer = response.content
    
    # Determine confidence based on top similarity score
    top_similarity = search_results[0]['similarity'] if search_results else 0
    if top_similarity > 0.8:
        confidence = "high"
    elif top_similarity > 0.6:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "query": query,
        "query_type": "question",
        "answer": answer,
        "citations": citations,
        "confidence": confidence
    }


def query_handler(query: str, bucket: str = BUCKET_NAME) -> Dict:
    """
    Main query handler - routes to appropriate mode.
    
    Args:
        query: User query
        bucket: S3 bucket name
        
    Returns:
        Response dictionary
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    # Load embeddings index
    print("\nLoading embeddings index from S3...")
    index_data = load_embeddings_index(bucket, EMBEDDINGS_INDEX_KEY)
    chunks = index_data['chunks']
    print(f"Loaded {len(chunks)} chunks")
    
    # Detect query type
    print("\nDetecting query type...")
    query_type, chapter_id = detect_query_type(query)
    print(f"Query type: {query_type.upper()}")
    
    if query_type == "citation":
        print(f"Chapter requested: {chapter_id}")
        response = handle_citation_mode(chunks, chapter_id, query)
    
    else:  # question mode
        print("Initializing models for Q&A mode...")
        embeddings_model = get_bedrock_embeddings()
        llm = get_bedrock_llm()
        response = handle_question_mode(query, chunks, embeddings_model, llm)
    
    print(f"\n{'='*80}")
    print("RESPONSE GENERATED")
    print(f"{'='*80}\n")
    
    return response


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "query": "What are medication management requirements?",
        "bucket_name": "medlaunch-rag"  # optional
    }
    """
    try:
        query = event.get('query')
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: query'
                })
            }
        
        bucket = event.get('bucket_name', BUCKET_NAME)
        
        response = query_handler(query, bucket)
        
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
        
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


def main():
    """Main function for local testing with interactive loop."""
    
    print("\n" + "="*80)
    print("🤖 NIAHO STANDARDS QUERY SYSTEM")
    print("="*80)
    print("\nWelcome! I can help you with NIAHO standards in two ways:")
    print("\n1. 📖 CITATION MODE - Get exact text from specific chapters")
    print("   Example: 'Show me chapter MM.1' or 'What does QM.1 say?'")
    print("\n2. 💬 Q&A MODE - Ask questions about standards")
    print("   Example: 'What are medication management requirements?'")
    print("\n" + "-"*80)
    print("Type 'exit' or 'quit' to stop the system")
    print("-"*80 + "\n")
    
    # Pre-load embeddings index once (optimization)
    print("🔄 Loading embeddings index from S3...")
    try:
        index_data = load_embeddings_index(BUCKET_NAME, EMBEDDINGS_INDEX_KEY)
        chunks = index_data['chunks']
        print(f"✅ Loaded {len(chunks)} chunks\n")
        
        # Pre-initialize models for Q&A mode (optimization)
        print("🔄 Initializing AI models...")
        embeddings_model = get_bedrock_embeddings()
        llm = get_bedrock_llm()
        print("✅ Models ready\n")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        print("Please ensure embeddings are generated first.\n")
        return
    
    # Interactive query loop
    query_count = 0
    
    while True:
        # Get user input
        try:
            user_input = input("🔍 Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
            print("\n👋 Thank you for using NIAHO Standards Query System. Goodbye!")
            break
        
        # Skip empty queries
        if not user_input:
            print("⚠️  Please enter a query.\n")
            continue
        
        query_count += 1
        
        print("\n" + "="*80)
        print(f"📝 QUERY #{query_count}: {user_input}")
        print("="*80)
        
        try:
            # Detect query type
            query_type, chapter_id = detect_query_type(user_input)
            print(f"\n🔎 Detected mode: {query_type.upper()}")
            
            if query_type == "citation":
                print(f"📌 Chapter requested: {chapter_id}")
                print("\n🔄 Retrieving exact text...\n")
                response = handle_citation_mode(chunks, chapter_id, user_input)
            
            else:  # question mode
                print("\n🔄 Searching knowledge base...")
                response = handle_question_mode(user_input, chunks, embeddings_model, llm)
            
            # Display response
            print("\n" + "-"*80)
            print("📋 RESPONSE:")
            print("-"*80 + "\n")
            
            if response['query_type'] == 'citation':
                if response.get('found') == False:
                    print(f"❌ {response['error']}")
                    print(f"💡 {response['suggestion']}\n")
                else:
                    print(f"📖 Chapter: {response['chapter']}")
                    print(f"📂 Section: {response['source']['section']}")
                    print(f"📄 Document: {response['source']['document']}")
                    print(f"\n{'─'*80}")
                    print("EXACT TEXT:")
                    print('─'*80)
                    print(f"\n{response['exact_text']}\n")
                    print('─'*80)
                    print(f"ℹ️  {response['disclaimer']}\n")
            
            else:  # question mode
                print(f"💡 Answer:\n\n{response['answer']}\n")
                print(f"📊 Confidence: {response['confidence'].upper()}")
                print(f"\n📚 Sources cited:")
                for i, citation in enumerate(response['citations'], 1):
                    print(f"   {i}. Chapter {citation['chapter']} - {citation['section']}")
                    print(f"      (Relevance: {citation['relevance_score']:.1%})")
                print()
            
            print("="*80 + "\n")
        
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\n" + "="*80 + "\n")
            continue


if __name__ == "__main__":
    main()