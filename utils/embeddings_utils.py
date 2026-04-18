"""
Local embeddings utility using sentence-transformers.
Replaces cloud-based IBM Watsonx embeddings with local model.
"""
from sentence_transformers import SentenceTransformer

from .singleton import Singleton


class EmbeddingClient(metaclass=Singleton):
    """
    Local embedding client using sentence-transformers.
    
    Uses 'sentence-transformers/all-MiniLM-L6-v2' model which produces
    384-dimensional embeddings (vs 1024-dim from Watsonx).
    
    The singleton pattern ensures the model is loaded only once per configuration.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding client with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (384-dim embeddings).
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def get_dense_embeddings(self, text_contents: list[str]) -> list[list[float]]:
        """
        Generate dense embeddings for a list of texts.
        
        Args:
            text_contents: List of text strings to embed.
        
        Returns:
            List of embedding vectors (each vector is a list of floats).
            For all-MiniLM-L6-v2, each vector has 384 dimensions.
        """
        # Generate embeddings using the sentence-transformers model
        embeddings = self.model.encode(
            text_contents,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convert numpy arrays to lists for compatibility
        return embeddings.tolist()
