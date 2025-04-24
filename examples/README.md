# Memory Module Examples

This directory contains example scripts demonstrating the usage of the `HybridSQLiteAdapter` and `EmbeddingManager` from the `src/memory` module.

These scripts serve as both usage examples and supplementary tests.

## Setup

1.  **Install Dependencies:** Ensure you have the necessary dependencies installed, particularly `sentence-transformers` for the embedding models used in these examples:
    ```bash
    pip install sentence-transformers
    ```
2.  **Project Root:** These scripts are designed to be run from the **project root directory** (`coding-agent/`) to ensure correct path resolution for imports and output files.

## Scripts

*   `memory/example_basic_usage.py`: 
    *   Demonstrates initializing the adapter and manager.
    *   Shows how to add `MemoryDoc` instances.
    *   Performs basic FTS (keyword) search using `adapter.query()`.
    *   Performs basic semantic (meaning-based) search using `adapter.semantic_query()`.
*   `memory/example_advanced_querying.py`: 
    *   Adds a more diverse set of documents with different tags and source agents.
    *   Demonstrates using filters with FTS search (`filter_tags`, `filter_source_agents`).
    *   Demonstrates using filters with semantic search (`filter_tags`, `filter_source_agents`).
    *   Shows combined filtering in both FTS and semantic search.

## Running the Examples

Navigate to the project root directory in your terminal and run the scripts using Python:

```bash
cd /path/to/coding-agent

# Run the basic example
python examples/memory/example_basic_usage.py

# Run the advanced querying example
python examples/memory/example_advanced_querying.py
```

## Output

The scripts will create an `output/` directory within the `examples/` directory to store the SQLite database files (`.db`) and embedding files (`.npy`). These files are cleaned up (removed) each time a script starts to ensure a fresh run.
