"""
Configuration file for MedlaunchRAG project.
Centralized configuration for all modules.
"""

import os
from typing import Dict

# ============================================================================
# AWS & S3 Configuration
# ============================================================================
AWS_REGION = 'us-east-1'
BUCKET_NAME = 'medlaunch-rag'

# ============================================================================
# PDF Processing Configuration
# ============================================================================
# Text splitter configuration
PDF_CHUNK_SIZE = 4000  # Characters per chunk for fallback chunking
PDF_CHUNK_OVERLAP = 200  # Character overlap between chunks

# Semantic splitting for large chapters
SEMANTIC_CHUNK_SIZE = 5000  # Characters (~1250 tokens)
SEMANTIC_CHUNK_OVERLAP = 300  # Character overlap for semantic chunks

# Token estimation settings
TOKENS_PER_CHAR = 4  # Approximate: 1 token ≈ 4 characters
MIN_CHAPTER_SIZE = 200  # Minimum characters to consider a chapter valid
MIN_TOKEN_COUNT = 500  # Minimum tokens for a chapter
MAX_TOKEN_COUNT = 1500  # Maximum tokens for single chunk

# ============================================================================
# Embedding Configuration
# ============================================================================
EMBEDDING_MODEL = 'amazon.titan-embed-text-v1'  # AWS Bedrock embedding model
EMBEDDING_DIMENSION = 1536  # Vector dimension for Titan embeddings

# S3 paths for chunks and embeddings
CHUNKS_PREFIX = 'chunks/'
EMBEDDINGS_PREFIX = 'embeddings/'
EMBEDDINGS_INDEX_KEY = 'embeddings/embeddings_index.json'

# Local storage
LOCAL_CHUNKS_DIR = './local_chunks'

# Batch processing
BATCH_SIZE = 25  # Number of chunks to embed in one API call (max for Titan is ~100)

# ============================================================================
# Query Handler Configuration
# ============================================================================
# LLM Configuration
LLM_MODEL = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'  # Inference profile ARN
LLM_TEMPERATURE = 0.1  # Lower temperature for more deterministic responses
LLM_MAX_TOKENS = 2000  # Maximum tokens in LLM response

# Vector search configuration
TOP_K_RESULTS = 5  # Number of top results to return from vector search

# Similarity thresholds for confidence scoring
SIMILARITY_HIGH_THRESHOLD = 0.8  # High confidence threshold
SIMILARITY_MEDIUM_THRESHOLD = 0.6  # Medium confidence threshold

# ============================================================================
# NIAHO Section Mapping
# ============================================================================
SECTION_MAPPING: Dict[str, str] = {
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

# ============================================================================
# File Paths (relative to project root)
# ============================================================================
FILES_DIR = './files'  # Input PDF files
CHUNKS_DIR = './chunks'  # Output chunks
EMBEDDINGS_FILE = './embeddings_index.json'  # Local embeddings index

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ============================================================================
# Feature Flags
# ============================================================================
# Enable local caching of chunks (set False to always download from S3)
ENABLE_LOCAL_CACHE = True

# Enable detailed logging during processing
VERBOSE_LOGGING = False
