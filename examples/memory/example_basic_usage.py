import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import shutil

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from memory.hybrid_sqlite import HybridSQLiteAdapter
from memory.embedding_manager import EmbeddingManager
from memory.adapter import MemoryDoc

# Attempt to import SentenceTransformer and handle potential ImportError
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: 'sentence-transformers' library not found.")
    print("Please install it using: pip install sentence-transformers")
    sys.exit(1)

# --- Configuration ---
DB_DIR = project_root / "examples" / "output"
DB_NAME = "basic_usage_memory.db"
EMBEDDING_DIR_NAME = "basic_usage_embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"  # Recommended: pip install sentence-transformers

DB_PATH = DB_DIR / DB_NAME
EMBEDDING_DIR = DB_DIR / EMBEDDING_DIR_NAME

def setup_environment():
    """Creates the output directory and cleans up old files."""
    print(f"Setting up environment in: {DB_DIR}")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        print(f"Removing existing database: {DB_PATH}")
        DB_PATH.unlink()
    if EMBEDDING_DIR.exists():
        print(f"Removing existing embedding directory: {EMBEDDING_DIR}")
        shutil.rmtree(EMBEDDING_DIR)
    print("Environment setup complete.")

def main():
    setup_environment()

    print(f"\n--- Initializing HybridSQLiteAdapter ---")
    print(f"Database Path: {DB_PATH}")
    print(f"Embedding Directory: {EMBEDDING_DIR}")
    print(f"Embedding Model: {MODEL_NAME}")

    # --- Instantiate Model ---
    print("\n--- Loading Embedding Model ---")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"\nError loading SentenceTransformer model '{MODEL_NAME}': {e}")
        print("Ensure the model name is correct and you have an internet connection for the first download.")
        return

    # --- Initialize Adapter ---
    print("\n--- Initializing HybridSQLiteAdapter ---")
    try:
        adapter = HybridSQLiteAdapter(
            db_path_str=str(DB_PATH),
            embedding_dir_str=str(EMBEDDING_DIR),
            embedding_model=model  # Pass the instantiated model object
        )
    except Exception as e:
        print(f"\nError initializing adapter: {e}")
        print("Please ensure 'sentence-transformers' is installed (`pip install sentence-transformers`) and the model is accessible.")
        return

    print("Adapter initialized successfully.")

    # --- Adding Documents ---
    print("\n--- Adding Documents ---")
    doc1 = MemoryDoc(
        id="doc_001",
        text="The quick brown fox jumps over the lazy dog.",
        timestamp=datetime.now(timezone.utc),
        source_agent="ExampleScript",
        tags=["animals", "proverb"],
        metadata={"source_file": "example_basic_usage.py"}
    )
    doc2 = MemoryDoc(
        id="doc_002",
        text="Large language models are trained on vast amounts of text data.",
        timestamp=datetime.now(timezone.utc),
        source_agent="ExampleScript",
        tags=["ai", "nlp"],
        metadata={"model_type": "transformer"}
    )
    doc3 = MemoryDoc(
        id="doc_003",
        text="Semantic search finds documents based on meaning, not just keywords.",
        timestamp=datetime.now(timezone.utc),
        source_agent="AnotherScript", # Different agent
        tags=["ai", "search"],
        metadata={"relevance": "high"}
    )

    adapter.add(doc1)
    print(f"Added document: {doc1.id} - '{doc1.text[:30]}...'" )
    adapter.add(doc2)
    print(f"Added document: {doc2.id} - '{doc2.text[:30]}...'" )
    adapter.add(doc3)
    print(f"Added document: {doc3.id} - '{doc3.text[:30]}...'" )

    # --- Performing Queries ---
    print("\n--- Performing Queries ---")

    # 1. Basic FTS Query
    query_fts = "language model"
    print(f"\n1. FTS Query: '{query_fts}'")
    results_fts = adapter.query(query_fts, k=5)
    print(f"Found {len(results_fts)} results (FTS):")
    for i, doc in enumerate(results_fts):
        print(f"  {i+1}. ID: {doc.id}, Score: {doc.score:.4f}, Text: '{doc.text[:50]}...'" )

    # 2. Basic Semantic Query
    query_semantic = "artificial intelligence techniques"
    print(f"\n2. Semantic Query: '{query_semantic}'")
    results_semantic = adapter.semantic_query(query_semantic, k=5)
    print(f"Found {len(results_semantic)} results (Semantic):")
    for i, doc in enumerate(results_semantic):
        print(f"  {i+1}. ID: {doc.id}, Score: {doc.score:.4f}, Text: '{doc.text[:50]}...'" )

    # --- Cleanup ---
    print("\n--- Closing Adapter ---")
    adapter.close()
    print("Adapter closed.")

if __name__ == "__main__":
    main()
