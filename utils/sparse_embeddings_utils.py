from pymilvus.model.sparse import BM25EmbeddingFunction

from .singleton import Singleton


class SparseEmbeddingClient(metaclass=Singleton):
    """Client for generating BM25 sparse embeddings for lexical search"""

    def __init__(self):
        self.bm25_ef = None
        self.corpus = []

    def fit_corpus(self, corpus: list[str]) -> None:
        """
        Fit the BM25 model on a corpus of documents.
        This must be called before generating embeddings.

        Args:
            corpus: List of text documents to fit the BM25 model
        """
        self.bm25_ef = BM25EmbeddingFunction()
        self.corpus = list(corpus)
        # Fit the BM25 model on the corpus
        self.bm25_ef.fit(self.corpus)

    def get_sparse_embeddings(self, texts: list[str]) -> list:
        """
        Generate BM25 sparse embeddings for given texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of sparse embeddings (one per text) compatible with Milvus
        """
        bm25_ef = self.bm25_ef
        if bm25_ef is None:
            raise ValueError("BM25 model not fitted. Call fit_corpus() first.")

        formatted_embeddings = []

        # Milvus expects a list of individual sparse matrices (one row each)
        for text in texts:
            row_matrix = bm25_ef.encode_documents([text])[0:1]
            formatted_embeddings.append(row_matrix)

        return formatted_embeddings

    def get_sparse_query_embedding(self, query: str):
        """
        Generate BM25 sparse embedding for a single query.

        Args:
            query: Query text string

        Returns:
            Sparse embedding as a single-row sparse matrix compatible with Milvus
        """
        bm25_ef = self.bm25_ef
        if bm25_ef is None:
            raise ValueError("BM25 model not fitted. Call fit_corpus() first.")

        # Generate sparse embedding for query - returns a multi-row sparse matrix
        sparse_matrix = bm25_ef.encode_queries([query])

        # Return the first (and only) row as a 1-row sparse matrix
        return sparse_matrix[0:1]

    def get_bm25_scores(self, query: str) -> list[float]:
        """
        Score the fitted corpus against a query using BM25 directly in Python.

        Args:
            query: Query text string

        Returns:
            BM25 score for each document in the fitted corpus
        """
        return self.get_bm25_scores_for_texts(query, self.corpus)

    def get_bm25_scores_for_texts(self, query: str, texts: list[str]) -> list[float]:
        """
        Score arbitrary texts against a query using the fitted BM25 model.

        Args:
            query: Query text string
            texts: Texts to score

        Returns:
            BM25 score for each provided text
        """
        bm25_ef = self.bm25_ef
        if bm25_ef is None:
            raise ValueError("BM25 model not fitted. Call fit_corpus() first.")

        if not texts:
            return []

        query_vector = bm25_ef.encode_queries([query])[0]
        document_vectors = bm25_ef.encode_documents(texts)

        scores = document_vectors.dot(query_vector.T).toarray().ravel()
        return scores.tolist()
