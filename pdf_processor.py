import re
import json
import boto3
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def download_pdf_from_s3(bucket: str, key: str, local_path: str) -> str:
    """
    Download PDF from S3 to local path.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local file path to save
        
    Returns:
        Local file path
    """
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)
    return local_path


def load_pdf_with_langchain(pdf_path: str) -> List[Document]:
    """
    Load PDF using LangChain's PyPDFLoader.
    
    Args:
        pdf_path: Local path to PDF file
        
    Returns:
        List of LangChain Document objects (one per page)
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# Section mapping from NIAHO table of contents
SECTION_MAPPING = {
    'QM': 'Quality Management System',
    'GB': 'Governing Body',
    'CE': 'Chief Executive Officer',
    'MS': 'Medical Staff',
    'NS': 'Nursing Services',
    'SM': 'Staffing Management',
    'MM': 'Medication Management',
    'SS': 'Surgical Services',
    'AS': 'Anesthesia Services',
    'OB': 'Obstetrical Care Services',
    'LS': 'Laboratory Services',
    'RC': 'Respiratory Care Services',
    'MI': 'Medical Imaging',
    'NM': 'Nuclear Medicine Services',
    'RS': 'Rehabilitation Services',
    'ES': 'Emergency Services',
    'OS': 'Outpatient Services',
    'DS': 'Dietary Services',
    'PR': 'Patient Rights',
    'IC': 'Infection Prevention and Control Program',
    'MR': 'Medical Records Service',
    'DC': 'Discharge Planning',
    'UR': 'Utilization Review',
    'PE': 'Physical Environment',
    'TO': 'Organ, Tissue and Eye Procurement',
    'SB': 'Swing Beds',
    'TD': 'Admission, Transfer and Discharge',
    'PC': 'Plan of Care',
    'RR': 'Residents Rights',
    'FS': 'Facility Services',
    'RN': 'Resident Nutrition',
    'PH-GR': 'Psychiatric Services - General Requirements',
    'PH-MR': 'Psychiatric Services - Medical Records',
    'PH-E': 'Psychiatric Services - Evaluation',
    'PH-NE': 'Psychiatric Services - Neurological Examination',
    'PH-TP': 'Psychiatric Services - Treatment Plan',
    'PH-PN': 'Psychiatric Services - Progress Notes',
    'PH-DP': 'Psychiatric Services - Discharge Planning',
    'PH-PR': 'Psychiatric Services - Personnel Resources',
    'PH-MS': 'Psychiatric Services - Medical Staff',
    'PH-NS': 'Psychiatric Services - Nursing Services',
    'PH-PS': 'Psychiatric Services - Psychological Services',
    'PH-SS': 'Psychiatric Services - Social Work Services',
    'PH-PA': 'Psychiatric Services - Psychosocial Assessment',
    'PH-TA': 'Psychiatric Services - Therapeutic Activities'
}


def extract_chapter_id(text: str) -> Optional[str]:
    """
    Extract chapter identifier from text (e.g., "QM.1", "LS.2", "PH-GR").
    
    Args:
        text: Text content to search for chapter ID
        
    Returns:
        Chapter ID if found, None otherwise
    """
    # Pattern for standard chapters (e.g., QM.1, MM.4)
    pattern1 = r'\b([A-Z]{2})\.(\d+)\b'
    # Pattern for psychiatric services (e.g., PH-GR, PH-MR)
    pattern2 = r'\b(PH-[A-Z]{2})\b'
    
    match = re.search(pattern1, text[:500])
    if match:
        return match.group(0)
    
    match = re.search(pattern2, text[:500])
    if match:
        return match.group(0)
    
    return None


def get_section_name(chapter_id: Optional[str]) -> str:
    """
    Get section name from chapter ID.
    
    Args:
        chapter_id: Chapter identifier (e.g., "QM.1", "PH-GR")
        
    Returns:
        Section name
    """
    if not chapter_id:
        return "General"
    
    # Handle psychiatric services (PH-XX)
    if chapter_id.startswith('PH-'):
        return SECTION_MAPPING.get(chapter_id, "Psychiatric Services")
    
    # Handle standard chapters (XX.N)
    prefix = chapter_id.split('.')[0]
    return SECTION_MAPPING.get(prefix, "General")


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.
    Approximation: 1 token ≈ 4 characters
    
    Args:
        text: Text content
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def find_chapter_boundaries(full_text: str) -> List[Tuple[int, str, str]]:
    """
    Find all chapter boundaries in the document.
    
    Args:
        full_text: Complete document text
        
    Returns:
        List of tuples (position, chapter_id, title)
    """
    chapters = []
    
    # Pattern 1: Standard chapters with titles (e.g., "QM.1 RESPONSIBILITY AND ACCOUNTABILITY")
    pattern1 = r'\n([A-Z]{2})\.(\d+)\s+([A-Z][A-Z\s,&\-()]+)\n'
    
    for match in re.finditer(pattern1, full_text):
        chapter_id = f"{match.group(1)}.{match.group(2)}"
        title = match.group(3).strip()
        position = match.start()
        
        chapters.append((position, chapter_id, title))
    
    # Pattern 2: Psychiatric services chapters (e.g., "GENERAL REQUIREMENTS (PH-GR)")
    pattern2 = r'\n([A-Z][A-Z\s]+)\(([A-Z]{2}-[A-Z]{2})\)'
    
    for match in re.finditer(pattern2, full_text):
        chapter_id = match.group(2)
        title = match.group(1).strip()
        position = match.start()
        
        chapters.append((position, chapter_id, title))
    
    # Sort by position
    chapters.sort(key=lambda x: x[0])
    
    return chapters


def extract_chapter_content(full_text: str, start_pos: int, end_pos: int) -> str:
    """
    Extract content between two chapter boundaries.
    
    Args:
        full_text: Complete document text
        start_pos: Start position
        end_pos: End position
        
    Returns:
        Chapter content
    """
    return full_text[start_pos:end_pos].strip()


def is_table_of_contents(text: str) -> bool:
    """
    Check if text is table of contents or similar non-content section.
    
    Args:
        text: Text to check
        
    Returns:
        True if TOC-like, False otherwise
    """
    toc_indicators = [
        'table of contents',
        'page ii of',
        'page iii of',
        'revision 25-1',
        'effective september'
    ]
    
    text_lower = text.lower()[:500]
    return any(indicator in text_lower for indicator in toc_indicators)


def chunk_by_chapters(documents: List[Document]) -> List[Dict]:
    """
    Chunk documents by chapter identifiers with smart logic.
    
    Args:
        documents: List of LangChain Document objects (pages from PDF)
        
    Returns:
        List of chunk dictionaries
    """
    # Combine all pages into single text
    full_text = "\n\n".join([doc.page_content for doc in documents])
    
    # Find all chapter boundaries
    print("Finding chapter boundaries...")
    chapter_boundaries = find_chapter_boundaries(full_text)
    print(f"Found {len(chapter_boundaries)} chapters")
    
    chunks = []
    
    if not chapter_boundaries:
        print("Warning: No chapters found, using fallback chunking")
        # Fallback to semantic chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )
        split_docs = text_splitter.create_documents([full_text])
        
        for doc in split_docs:
            chunks.append({
                'text': doc.page_content,
                'chapter_id': None,
                'section_name': "General"
            })
        return chunks
    
    # Process each chapter
    for i, (position, chapter_id, title) in enumerate(chapter_boundaries):
        # Determine end position
        if i + 1 < len(chapter_boundaries):
            end_position = chapter_boundaries[i + 1][0]
        else:
            end_position = len(full_text)
        
        # Extract chapter content
        chapter_text = extract_chapter_content(full_text, position, end_position)
        
        # Skip table of contents and similar sections
        if is_table_of_contents(chapter_text):
            print(f"Skipping TOC-like content: {chapter_id}")
            continue
        
        # Skip very small chapters (likely false positives)
        if len(chapter_text) < 200:
            print(f"Skipping small chapter: {chapter_id} ({len(chapter_text)} chars)")
            continue
        
        token_count = estimate_token_count(chapter_text)
        
        # If chapter is within optimal range, keep as single chunk
        if 500 <= token_count <= 1500:
            chunks.append({
                'text': chapter_text,
                'chapter_id': chapter_id,
                'section_name': get_section_name(chapter_id),
                'title': title
            })
            print(f"  ✓ {chapter_id}: {token_count} tokens (single chunk)")
        
        # If chapter is too large, split it intelligently
        elif token_count > 1500:
            print(f"  ! {chapter_id}: {token_count} tokens (splitting...)")
            
            # Try to split by subsections first
            subsection_pattern = r'\n([A-Z][A-Z\s]+):\s*\n'
            subsections = re.split(subsection_pattern, chapter_text)
            
            if len(subsections) > 3:
                # We found subsections, use them
                for j in range(1, len(subsections), 2):
                    if j + 1 < len(subsections):
                        subsection_title = subsections[j].strip()
                        subsection_content = subsections[j + 1].strip()
                        combined = f"{subsection_title}:\n{subsection_content}"
                        
                        if len(combined) > 200:
                            chunks.append({
                                'text': combined,
                                'chapter_id': chapter_id,
                                'section_name': get_section_name(chapter_id),
                                'title': f"{title} - {subsection_title}"
                            })
                print(f"    → Split into {(len(subsections)-1)//2} subsections")
            else:
                # No clear subsections, use semantic splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000,  # ~1250 tokens
                    chunk_overlap=300,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " "]
                )
                sub_chunks = text_splitter.create_documents([chapter_text])
                
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'text': sub_chunk.page_content,
                        'chapter_id': chapter_id,
                        'section_name': get_section_name(chapter_id),
                        'title': title
                    })
                print(f"    → Split into {len(sub_chunks)} semantic chunks")
        
        # If chapter is too small but valid, keep it
        else:
            chunks.append({
                'text': chapter_text,
                'chapter_id': chapter_id,
                'section_name': get_section_name(chapter_id),
                'title': title
            })
            print(f"  ✓ {chapter_id}: {token_count} tokens (kept small chunk)")
    
    return chunks


def create_chunk_objects(chunks: List[Dict], document_name: str = "NIAHO Standards") -> List[Dict]:
    """
    Create formatted chunk objects with metadata.
    
    Args:
        chunks: List of raw chunks
        document_name: Name of the document
        
    Returns:
        List of formatted chunk objects
    """
    chunk_objects = []
    
    for idx, chunk in enumerate(chunks, start=1):
        chunk_obj = {
            "chunk_id": f"{idx:03d}",
            "text": chunk['text'],
            "metadata": {
                "document": document_name,
                "section": chunk['section_name'],
                "chapter": chunk['chapter_id'] if chunk['chapter_id'] else "N/A",
                "title": chunk.get('title', '')
            },
            "token_count": estimate_token_count(chunk['text']),
            "embedding": None
        }
        chunk_objects.append(chunk_obj)
    
    return chunk_objects


def save_chunks_to_s3(chunks: List[Dict], bucket: str, prefix: str = 'chunks/'):
    """
    Save chunks to S3 as JSON files.
    
    Args:
        chunks: List of chunk objects
        bucket: S3 bucket name
        prefix: S3 prefix for chunks
    """
    s3 = boto3.client('s3')
    
    for chunk in chunks:
        chunk_key = f"{prefix}chunk_{chunk['chunk_id']}.json"
        s3.put_object(
            Bucket=bucket,
            Key=chunk_key,
            Body=json.dumps(chunk, indent=2),
            ContentType='application/json'
        )


def main():
    
    # Configuration
    BUCKET_NAME = 'medlaunch-rag'
    PDF_KEY = 'raw/niaho_standards.pdf'
    LOCAL_PATH = './downloaded_niaho.pdf'
    
    print("=" * 80)
    print("OPTIMIZED CHAPTER-BASED CHUNKING")
    print("=" * 80)
    
    # Step 1: Download and load PDF
    print("\nStep 1: Downloading PDF from S3...")
    local_file = download_pdf_from_s3(BUCKET_NAME, PDF_KEY, LOCAL_PATH)
    
    print("\nStep 2: Loading PDF with LangChain...")
    documents = load_pdf_with_langchain(local_file)
    print(f"Loaded {len(documents)} pages")
    
    # Step 3: Chunk by chapters
    print("\nStep 3: Chunking by chapters...")
    raw_chunks = chunk_by_chapters(documents)
    print(f"\nCreated {len(raw_chunks)} chapter-based chunks")
    
    # Step 4: Create formatted chunk objects
    print("\nStep 4: Creating formatted chunk objects...")
    chunk_objects = create_chunk_objects(raw_chunks)
    
    # Display statistics
    token_counts = [c['token_count'] for c in chunk_objects]
    print(f"\n📊 Chunk Statistics:")
    print(f"  Total chunks: {len(chunk_objects)}")
    print(f"  Token count range: {min(token_counts)} - {max(token_counts)}")
    print(f"  Average tokens: {sum(token_counts) // len(token_counts)}")
    print(f"  Median tokens: {sorted(token_counts)[len(token_counts)//2]}")
    
    # Count chunks by section
    sections = {}
    for chunk in chunk_objects:
        section = chunk['metadata']['section']
        sections[section] = sections.get(section, 0) + 1
    
    print(f"\n📋 Chunks by Section:")
    for section, count in sorted(sections.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {section}: {count} chunks")
    
    # Display sample chunks
    print(f"\n📄 Sample Chunks:")
    for chunk in chunk_objects[:5]:
        print(f"\n  Chunk ID: {chunk['chunk_id']}")
        print(f"  Section: {chunk['metadata']['section']}")
        print(f"  Chapter: {chunk['metadata']['chapter']}")
        print(f"  Title: {chunk['metadata']['title']}")
        print(f"  Token Count: {chunk['token_count']}")
        print(f"  Text preview: {chunk['text'][:100]}...")
    
    # Step 5: Save to S3
    print(f"\n\nStep 5: Saving {len(chunk_objects)} chunks to S3...")
    save_chunks_to_s3(chunk_objects, BUCKET_NAME)
    print("✅ All chunks saved to S3")
    
    print("\n" + "=" * 80)
    print(f"✅ CHUNKING COMPLETE - {len(chunk_objects)} chunks created")
    print("=" * 80)


if __name__ == "__main__":
    main()