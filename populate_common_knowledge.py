import os
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create the common knowledge collection
collection = client.get_or_create_collection(name="common_knowledge")

# Load the Indian-Law dataset
logger.info("Loading Indian-Law dataset...")
dataset = load_dataset("vishnun0027/Indian-Law")

# Verify dataset structure
logger.info(f"Dataset columns: {dataset['train'].column_names}")

# Process the dataset in batches
batch_size = 100
total_processed = 0
total_skipped = 0

for i in range(0, len(dataset['train']), batch_size):
    logger.info(f"Processing batch {i // batch_size + 1}...")
    batch = dataset['train'][i:i + batch_size]
    responses = []
    instructions = []
    valid_indices = []

    # Filter valid entries
    for j, (resp, instr) in enumerate(zip(batch['Response'], batch['Instruction'])):
        if isinstance(resp, str) and resp.strip() and isinstance(instr, str) and instr.strip():
            responses.append(resp)
            instructions.append(instr)
            valid_indices.append(j)
        else:
            logger.warning(f"Skipping entry at index {i + j}: Response or Instruction invalid (Response: {resp}, Instruction: {instr})")
            total_skipped += 1

    if not responses:
        logger.info("No valid entries in this batch, skipping...")
        continue

    # Generate embeddings
    try:
        embeddings_batch = embeddings.embed_documents(responses)
    except Exception as e:
        logger.error(f"Error embedding batch {i // batch_size + 1}: {e}")
        continue

    # Prepare metadata and IDs
    metadatas = [{'source': 'Indian Law Dataset', 'question': instr} for instr in instructions]
    ids = [f"common_{i + j}" for j in valid_indices]

    # Insert into ChromaDB
    try:
        collection.add(
            documents=responses,
            embeddings=embeddings_batch,
            metadatas=metadatas,
            ids=ids
        )
        total_processed += len(responses)
        logger.info(f"Added {len(responses)} documents to common_knowledge collection.")
    except Exception as e:
        logger.error(f"Error adding batch {i // batch_size + 1} to ChromaDB: {e}")

logger.info(f"Completed. Total documents processed: {total_processed}, Total skipped: {total_skipped}")