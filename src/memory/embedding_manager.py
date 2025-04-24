import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_distance
import filelock

log = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation, storage, retrieval, and similarity calculations."""

    def __init__(self, embedding_model: SentenceTransformer | None, embedding_dir: str | Path):
        self.embedding_model = embedding_model
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        if self.embedding_model is None:
            log.warning("Embedding model not provided to EmbeddingManager. Semantic search features will be disabled.")

    def _get_embedding_path(self, doc_id: str) -> Path:
        """Constructs the absolute path for an embedding file."""
        return self.embedding_dir / f"{doc_id}.npy"

    def generate_embedding(self, text: str) -> np.ndarray | None:
        """Generates an embedding for the given text."""
        if not self.embedding_model or not text:
            return None
        try:
            log.debug("Generating embedding.")
            # Normalize embeddings for cosine similarity
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            log.debug(f"Generated embedding with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            log.error(f"Error generating embedding: {e}", exc_info=True)
            return None

    def save_embedding(self, embedding: np.ndarray, doc_id: str) -> Path | None:
        """Saves an embedding array to a .npy file with locking."""
        if embedding is None or not doc_id:
            return None

        embedding_abs_path = self._get_embedding_path(doc_id)
        lock_path = embedding_abs_path.with_suffix(".npy.lock")

        try:
            lock = filelock.FileLock(str(lock_path))
            with lock.acquire(timeout=5):
                log.debug(f"Saving embedding to {embedding_abs_path}")
                np.save(embedding_abs_path, embedding)
                log.debug(f"Saved embedding for doc_id: {doc_id}")
                return embedding_abs_path
        except filelock.Timeout:
            log.error(f"Timeout acquiring lock for embedding file {lock_path} (doc_id: {doc_id})")
            return None
        except Exception as e:
            log.error(f"Error saving embedding to {embedding_abs_path} (doc_id: {doc_id}): {e}", exc_info=True)
            return None
        finally:
            # Clean up lock file if it exists and wasn't automatically cleaned
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError as e:
                    log.warning(f"Could not clean up lock file {lock_path}: {e}")

    def load_embedding(self, embedding_path: str | Path) -> np.ndarray | None:
        """Loads an embedding from a .npy file."""
        if not embedding_path:
            return None
        try:
            path = Path(embedding_path)
            if path.exists():
                log.debug(f"Loading embedding from {path}")
                return np.load(path)
            else:
                log.warning(f"Embedding file not found: {path}")
                return None
        except Exception as e:
            log.error(f"Error loading embedding from {embedding_path}: {e}", exc_info=True)
            return None

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculates cosine similarity between two embeddings."""
        # cosine_distance = 1 - cosine_similarity
        similarity = 1 - cosine_distance(emb1, emb2)
        return float(similarity) # Ensure result is standard float

    def generate_and_save_embedding(self, text: str, doc_id: str) -> Path | None:
        """Convenience method to generate and save an embedding."""
        embedding = self.generate_embedding(text)
        if embedding is not None:
            return self.save_embedding(embedding, doc_id)
        return None
