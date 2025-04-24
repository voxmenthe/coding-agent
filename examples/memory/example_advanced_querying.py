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
DB_NAME = "advanced_querying_memory.db"
EMBEDDING_DIR_NAME = "advanced_querying_embeddings"
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

    print(f"\n--- Loading Embedding Model ---")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"\nError loading SentenceTransformer model '{MODEL_NAME}': {e}")
        print("Ensure the model name is correct and you have an internet connection for the first download.")
        return

    print(f"\n--- Initializing HybridSQLiteAdapter ---")
    print(f"Database Path: {DB_PATH}")
    print(f"Embedding Directory: {EMBEDDING_DIR}")
    print(f"Embedding Model: {MODEL_NAME}")

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

    # --- Adding Diverse Documents ---
    print("\n--- Adding Diverse Documents ---")
    docs_data = [
        {"id": "proj_idea_01", "text": "Develop a system for real-time sentiment analysis of social media feeds.", "tags": ["ai", "nlp", "project"], "source_agent": "BrainstormBot"},
        {"id": "meeting_notes_01", "text": "Discussed project timelines. Q3 goals involve finalizing the API.", "tags": ["meeting", "planning", "api"], "source_agent": "MeetingSummarizer"},
        {"id": "code_snippet_01", "text": "Use pandas for data manipulation and scikit-learn for ML models.", "tags": ["python", "code", "ml"], "source_agent": "CodeHelper"},
        {"id": "proj_idea_02", "text": "Create an AI-powered chatbot for customer support.", "tags": ["ai", "nlp", "project", "chatbot"], "source_agent": "BrainstormBot"},
        {"id": "tech_article_01", "text": "Attention mechanisms are crucial for transformer model performance.", "tags": ["ai", "nlp", "transformer"], "source_agent": "ResearchAgent"},
        {"id": "meeting_notes_02", "text": "Agreed on using Python for the backend API development.", "tags": ["meeting", "decision", "python", "api"], "source_agent": "MeetingSummarizer"},
    ]

    for data in docs_data:
        doc = MemoryDoc(
            id=data["id"],
            text=data["text"],
            timestamp=datetime.now(timezone.utc),
            source_agent=data["source_agent"],
            tags=data["tags"],
            metadata={"example_source": "advanced_querying.py"}
        )
        adapter.add(doc)
        print(f"Added document: {doc.id} - Tags: {doc.tags}, Agent: {doc.source_agent}")

    # --- Performing Advanced Queries ---
    print("\n--- Performing Advanced Queries ---")

    # 1. FTS Query with Tag Filter
    query_fts_tags = "API development"
    filter_tags_fts = ["api", "python"]
    print(f"\n1. FTS Query: '{query_fts_tags}', Filter Tags: {filter_tags_fts}")
    try:
        results_fts_tags = adapter.query(query_fts_tags, k=5, filter_tags=filter_tags_fts)
        print(f"Found {len(results_fts_tags)} results:")
        for i, doc in enumerate(results_fts_tags):
            print(f"  {i+1}. ID: {doc.id}, Score: {doc.score:.4f}, Tags: {doc.tags}, Text: '{doc.text[:50]}...'" )
    except Exception as e:
        print(f"\nError performing FTS query with tag filter: {e}")

    # 2. FTS Query with Source Agent Filter
    query_fts_agent = "project discussion"
    filter_agent_fts = "MeetingSummarizer"
    print(f"\n2. FTS Query: '{query_fts_agent}', Filter Agent: {filter_agent_fts}")
    try:
        results_fts_agent = adapter.query(query_fts_agent, k=5, filter_source_agents=[filter_agent_fts])
        print(f"Found {len(results_fts_agent)} results:")
        for i, doc in enumerate(results_fts_agent):
            print(f"  {i+1}. ID: {doc.id}, Score: {doc.score:.4f}, Agent: {doc.source_agent}, Text: '{doc.text[:50]}...'" )
    except Exception as e:
        print(f"\nError performing FTS query with agent filter: {e}")

    # 3. Semantic Query with Tag Filter
    query_semantic_tags = "machine learning code examples"
    filter_tags_semantic = ["python", "ml"]
    print(f"\n3. Semantic Query: '{query_semantic_tags}', Filter Tags: {filter_tags_semantic}")
    try:
        results_semantic_tags = adapter.semantic_query(query_semantic_tags, k=5, filter_tags=filter_tags_semantic)
        print(f"Found {len(results_semantic_tags)} results:")
        for i, doc in enumerate(results_semantic_tags):
            print(f"  {i+1}. ID: {doc.id}, Score: {doc.score:.4f}, Tags: {doc.tags}, Text: '{doc.text[:50]}...'" )
    except Exception as e:
        print(f"\nError performing semantic query with tag filter: {e}")

    # 4. Semantic Query with Source Agent Filter
    query_semantic_agent = "ideas about natural language processing"
    filter_agents_semantic = ["BrainstormBot"]
    print(f"\n4. Semantic Query: '{query_semantic_agent}', Filter Agents: {filter_agents_semantic}")
    try:
        results_semantic_agent = adapter.semantic_query(query_semantic_agent, k=5, filter_source_agents=filter_agents_semantic)
        print(f"Found {len(results_semantic_agent)} results:")
        for i, doc in enumerate(results_semantic_agent):
            print(f"  {i+1}. ID: {doc.id}, Score: {doc.score:.4f}, Agent: {doc.source_agent}, Text: '{doc.text[:50]}...'" )
    except Exception as e:
        print(f"\nError performing semantic query with agent filter: {e}")

    # 5. Semantic Query with Combined Filters
    query_semantic_combined = "python api plans"
    filter_tags_combined = ["api"]
    filter_agents_combined = ["MeetingSummarizer"]
    print(f"\n5. Semantic Query: '{query_semantic_combined}', Filter Tags: {filter_tags_combined}, Filter Agents: {filter_agents_combined}")
    try:
        results_semantic_combined = adapter.semantic_query(query_semantic_combined, k=5, filter_tags=filter_tags_combined, filter_source_agents=filter_agents_combined)
        print(f"Found {len(results_semantic_combined)} results:")
        for i, doc in enumerate(results_semantic_combined):
            print(f"  {i+1}. ID: {doc.id}, Score: {doc.score:.4f}, Tags: {doc.tags}, Agent: {doc.source_agent}, Text: '{doc.text[:50]}...'" )
    except Exception as e:
        print(f"\nError performing semantic query with combined filters: {e}")


    # --- Cleanup ---
    print("\n--- Closing Adapter ---")
    adapter.close()
    print("Adapter closed.")

if __name__ == "__main__":
    main()
