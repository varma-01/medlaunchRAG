import json
import boto3
import os
from typing import List, Dict
from langchain_aws import BedrockEmbeddings

from config import (
    AWS_REGION,
    BUCKET_NAME,
    CHUNKS_PREFIX,
    EMBEDDINGS_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    LOCAL_CHUNKS_DIR,
    BATCH_SIZE,
)


def get_s3_client():
    """Initialize S3 client."""
    return boto3.client('s3')


def get_bedrock_embeddings():
    """
    Initialize Bedrock embeddings using Amazon Titan.
    
    Returns:
        BedrockEmbeddings instance
    """
    return BedrockEmbeddings(
        model_id=EMBEDDING_MODEL,
        region_name=AWS_REGION
    )


def download_all_chunks(bucket: str, prefix: str, local_dir: str) -> List[str]:
    """
    Download all chunks from S3 to local directory.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix for chunks
        local_dir: Local directory to save chunks
        
    Returns:
        List of local file paths
    """
    s3 = get_s3_client()
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # List all chunks
    local_files = []
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    print(f"Downloading chunks from S3...")
    chunk_count = 0
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.json') and key != prefix:
                    # Download file
                    filename = os.path.basename(key)
                    local_path = os.path.join(local_dir, filename)
                    
                    s3.download_file(bucket, key, local_path)
                    local_files.append(local_path)
                    chunk_count += 1
    
    print(f"Downloaded {chunk_count} chunks to {local_dir}")
    return local_files


def load_chunks_from_local(local_files: List[str]) -> List[Dict]:
    """
    Load all chunk JSONs from local files.
    
    Args:
        local_files: List of local file paths
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    for file_path in local_files:
        with open(file_path, 'r') as f:
            chunk_data = json.load(f)
            chunks.append(chunk_data)
    
    return chunks


def generate_embeddings_batch(chunks: List[Dict], embeddings_model: BedrockEmbeddings, batch_size: int = 20) -> List[Dict]:
    """
    Generate embeddings for all chunks using batch processing.
    
    Args:
        chunks: List of chunk dictionaries
        embeddings_model: BedrockEmbeddings instance
        batch_size: Number of texts to embed in one API call
        
    Returns:
        List of chunks with embeddings added
    """
    print(f"\nGenerating embeddings for {len(chunks)} chunks in batches of {batch_size}...")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(chunks), batch_size):
        batch_end = min(batch_idx + batch_size, len(chunks))
        current_batch = chunks[batch_idx:batch_end]
        
        batch_num = (batch_idx // batch_size) + 1
        print(f"  Processing batch {batch_num}/{total_batches} (chunks {batch_idx+1}-{batch_end})...")
        
        # Extract texts from current batch
        texts = [chunk['text'] for chunk in current_batch]
        
        # Generate embeddings for entire batch in one API call
        try:
            embeddings = embeddings_model.embed_documents(texts)
            
            # Assign embeddings back to chunks
            for i, chunk in enumerate(current_batch):
                chunk['embedding'] = embeddings[i]
            
            print(f"    ✓ Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            print(f"    ✗ Error in batch {batch_num}: {str(e)}")
            print(f"    Falling back to individual embedding for this batch...")
            
            # Fallback: embed individually if batch fails
            for chunk in current_batch:
                try:
                    embedding = embeddings_model.embed_query(chunk['text'])
                    chunk['embedding'] = embedding
                except Exception as e2:
                    print(f"      ✗ Failed to embed chunk {chunk['chunk_id']}: {str(e2)}")
                    chunk['embedding'] = [0.0] * EMBEDDING_DIMENSION  # Zero vector as fallback
    
    print(f"✓ Generated embeddings for all chunks")
    return chunks


# def save_chunks_locally(chunks: List[Dict], local_dir: str):
#     """
#     Save updated chunks with embeddings to local files.
    
#     Args:
#         chunks: List of chunk dictionaries with embeddings
#         local_dir: Local directory to save chunks
#     """
#     for chunk in chunks:
#         filename = f"chunk_{chunk['chunk_id']}.json"
#         file_path = os.path.join(local_dir, filename)
        
#         with open(file_path, 'w') as f:
#             json.dump(chunk, f, indent=2)


def create_embeddings_index(chunks: List[Dict]) -> Dict:
    """
    Create consolidated embeddings index with all chunks.
    
    Args:
        chunks: List of all chunk objects with embeddings
        
    Returns:
        Embeddings index dictionary
    """
    index = {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "total_chunks": len(chunks),
        "chunks": chunks
    }
    return index


# def upload_all_chunks(local_dir: str, bucket: str, prefix: str):
#     """
#     Upload all updated chunks back to S3.
    
#     Args:
#         local_dir: Local directory containing chunks
#         bucket: S3 bucket name
#         prefix: S3 prefix for chunks
#     """
#     s3 = get_s3_client()
    
#     print(f"\nUploading updated chunks to S3...")
    
#     files = [f for f in os.listdir(local_dir) if f.endswith('.json')]
    
#     for filename in files:
#         local_path = os.path.join(local_dir, filename)
#         s3_key = f"{prefix}{filename}"
        
#         with open(local_path, 'r') as f:
#             chunk_data = json.load(f)
        
#         s3.put_object(
#             Bucket=bucket,
#             Key=s3_key,
#             Body=json.dumps(chunk_data, indent=2),
#             ContentType='application/json'
#         )
    
#     print(f"✓ Uploaded {len(files)} chunks to S3")


def upload_embeddings_index(index: Dict, bucket: str):
    """
    Upload embeddings index to S3.
    
    Args:
        index: Embeddings index dictionary
        bucket: S3 bucket name
    """
    s3 = get_s3_client()
    index_key = f"{EMBEDDINGS_PREFIX}embeddings_index.json"
    
    s3.put_object(
        Bucket=bucket,
        Key=index_key,
        Body=json.dumps(index, indent=2),
        ContentType='application/json'
    )
    
    print(f"✓ Uploaded embeddings_index.json to S3")


def cleanup_local_files(local_dir: str):
    """
    Clean up local chunk files.
    
    Args:
        local_dir: Local directory to clean
    """
    import shutil
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
        print(f"✓ Cleaned up local directory: {local_dir}")


def generate_embeddings(bucket: str):
    """
    Main function to generate embeddings for all chunks.
    
    Args:
        bucket: S3 bucket name
        
    Returns:
        Summary dictionary with results
    """
    print("=" * 80)
    print("Generating EMBEDDINGS")
    print("=" * 80)
    
    # Step 1: Download all chunks from S3
    print("\nDownloading all chunks from S3...")
    local_files = download_all_chunks(bucket, CHUNKS_PREFIX, LOCAL_CHUNKS_DIR)
    
    if not local_files:
        print("No chunks found. Run pdf_processor first.")
        return {"status": "error", "message": "No chunks found"}
    
    # Step 2: Load all chunks into memory
    print("\nLoading chunks from local files...")
    chunks = load_chunks_from_local(local_files)
    print(f"Loaded {len(chunks)} chunks")
    
    # Step 3: Generate embeddings for all chunks
    print("\nGenerating embeddings with Amazon Titan (batch mode)...")
    embeddings_model = get_bedrock_embeddings()
    chunks_with_embeddings = generate_embeddings_batch(chunks, embeddings_model, batch_size=BATCH_SIZE)
    
    # # Step 4: Save updated chunks locally
    # print("\nStep 4: Saving updated chunks locally...")
    # save_chunks_locally(chunks_with_embeddings, LOCAL_CHUNKS_DIR)
    
    # Step 5: Create embeddings index
    print("\nStep 5: Creating embeddings index...")
    embeddings_index = create_embeddings_index(chunks_with_embeddings)
    print(f"Created index with {len(chunks_with_embeddings)} chunks")
    
    # # Step 6: Upload all chunks back to S3
    # print("\nStep 6: Uploading updated chunks to S3...")
    # upload_all_chunks(LOCAL_CHUNKS_DIR, bucket, CHUNKS_PREFIX)
    
    # Step 7: Upload embeddings index
    print("\nStep 7: Uploading embeddings index to S3...")
    upload_embeddings_index(embeddings_index, bucket)
    
    # Step 8: Cleanup local files
    print("\nStep 8: Cleaning up local files...")
    cleanup_local_files(LOCAL_CHUNKS_DIR)
    
    # Summary
    summary = {
        "status": "success",
        "total_chunks": len(chunks_with_embeddings),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embeddings_index_key": f"{EMBEDDINGS_PREFIX}embeddings_index.json"
    }
    
    print("\n" + "=" * 80)
    print("✅ EMBEDDING GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total chunks processed: {summary['total_chunks']}")
    print(f"  Embedding model: {summary['embedding_model']}")
    print(f"  Embedding dimension: {summary['embedding_dimension']}")
    print(f"  Index location: s3://{bucket}/{summary['embeddings_index_key']}")
    
    return summary


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "bucket_name": "medlaunch-rag"  # optional
    }
    """
    try:
        bucket = event.get('bucket_name', BUCKET_NAME)
        result = generate_embeddings(bucket)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }


def main():
    """Main function for local testing."""
    result = generate_embeddings(BUCKET_NAME)
    # print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()