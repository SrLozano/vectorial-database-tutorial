import logging
import time
from contextlib import contextmanager
from typing import Any

from pymilvus import CollectionSchema
from pymilvus import MilvusClient as MC
from pymilvus.exceptions import MilvusException

from .embeddings_utils import EmbeddingClient
from .singleton import Singleton
from .sparse_embeddings_utils import SparseEmbeddingClient


class MilvusClient(metaclass=Singleton):
    """
    Milvus client using Milvus Lite for local, file-based vector storage.

    This replaces the cloud-based Milvus connection with a local database file.
    """

    def __init__(
        self,
        db_path: str = "./milvus_demo.db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize Milvus Lite client with local file storage.

        Args:
            db_path: Path to the local Milvus Lite database file.
            model_name: Name of the sentence-transformers model for embeddings.
        """
        # Use Milvus Lite with local file path
        self.client = MC(uri=db_path)

        # Initialize embedding clients
        self.embeddings = EmbeddingClient(model_name=model_name)
        self.sparse_embeddings = SparseEmbeddingClient()

    def create_collection(
        self, collection_name: str, schema: CollectionSchema | None = None, **kwargs
    ) -> None:
        if not self.client.has_collection(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name, schema=schema, **kwargs
            )
        self._wait_for_collection_ready(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        self.client.drop_collection(collection_name)

    def list_collections(self) -> Any:
        return self.client.list_collections()

    def flush_collection(self, collection_name: str) -> None:
        self.client.flush(collection_name=collection_name)

    def load_collection(self, collection_name: str) -> None:
        """Load collection into memory for search operations"""
        self.client.load_collection(collection_name=collection_name)

    def build_index(
        self, collection_name: str, index_type: str, field_name: str, metric_type: str
    ) -> None:
        if not self.client.has_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        if self._field_index_exists(collection_name, field_name):
            return

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=field_name, metric_type=metric_type, index_type=index_type
        )
        self._create_index_with_retry(
            collection_name=collection_name, index_params=index_params
        )

    def _wait_for_collection_ready(
        self, collection_name: str, timeout: float = 10.0, interval: float = 0.25
    ) -> None:
        deadline = time.time() + timeout
        last_error: Exception | None = None

        while time.time() < deadline:
            try:
                if self.client.has_collection(collection_name=collection_name):
                    self.client.describe_collection(collection_name=collection_name)
                    return
            except MilvusException as exc:
                last_error = exc

            time.sleep(interval)

        if last_error is not None:
            raise last_error

        raise TimeoutError(
            f"Collection '{collection_name}' was not ready in {timeout}s"
        )

    def _field_index_exists(self, collection_name: str, field_name: str) -> bool:
        return bool(self.client.list_indexes(collection_name, field_name=field_name))

    def _create_index_with_retry(
        self,
        collection_name: str,
        index_params: Any,
        retries: int = 5,
        base_delay: float = 0.5,
    ) -> None:
        last_error: MilvusException | None = None

        for attempt in range(retries):
            try:
                with self._suppress_milvus_logs():
                    self.client.create_index(
                        collection_name=collection_name, index_params=index_params
                    )
                return
            except MilvusException as exc:
                last_error = exc
                error_text = str(exc).lower()

                if (
                    not self._is_transient_index_error(error_text)
                    or attempt == retries - 1
                ):
                    raise

                time.sleep(base_delay * (attempt + 1))
                self._wait_for_collection_ready(collection_name)

        if last_error is not None:
            raise last_error

    @staticmethod
    def _is_transient_index_error(error_text: str) -> bool:
        return "internal error" in error_text or "unrecognized token" in error_text

    @staticmethod
    @contextmanager
    def _suppress_milvus_logs():
        loggers = [
            logging.getLogger("pymilvus.decorators"),
            logging.getLogger("pymilvus.client.grpc_handler"),
        ]
        previous_states = [(logger.level, logger.disabled) for logger in loggers]

        try:
            for logger in loggers:
                logger.disabled = True
            yield
        finally:
            for logger, (level, disabled) in zip(loggers, previous_states):
                logger.setLevel(level)
                logger.disabled = disabled

    def insert_data(
        self, documents: dict | list[dict], collection_name: str
    ) -> dict[str, Any]:
        insertion = self.client.insert(collection_name=collection_name, data=documents)
        return insertion

    def upsert(self, collection_name: str, documents: dict | list[dict]) -> None:
        self.client.upsert(collection_name=collection_name, data=documents)

    def query(
        self,
        collection_name: str,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] = ["*"],
    ) -> list[dict]:
        return self.client.query(
            collection_name=collection_name,
            filter=filter,
            limit=limit,
            output_fields=output_fields,
        )

    def lexical_search(
        self,
        query: str,
        collection_name: str,
        output_fields: list[str],
        limit=10,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        filter_expression = kwargs.get("filter", "")
        text_field = kwargs.get("text_field", "text")
        fields_to_fetch = list(dict.fromkeys([*output_fields, text_field]))

        query_results = self.query(
            collection_name=collection_name,
            filter=filter_expression,
            limit=16383,
            output_fields=fields_to_fetch,
        )

        texts = [str(entity.get(text_field, "")) for entity in query_results]
        scores = self.sparse_embeddings.get_bm25_scores_for_texts(query, texts)

        scored_results = []
        for entity, score in zip(query_results, scores):
            if score <= 0:
                continue

            scored_results.append(
                {
                    "id": entity.get("id"),
                    "distance": score,
                    "entity": entity,
                }
            )

        scored_results.sort(key=lambda item: item["distance"], reverse=True)
        return [scored_results[:limit]]

    def semantic_search(
        self,
        collection_name: str,
        query: str,
        output_fields: list[str],
        anns_field: str,
        limit=10,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        filter_expression = kwargs.get("filter", "")
        search_params = {
            "metric_type": "COSINE",
        }

        dense_vector_query = self.get_dense_vector(query)

        res = self.client.search(
            collection_name=collection_name,
            data=[dense_vector_query],
            filter=filter_expression,
            anns_field=anns_field,
            output_fields=output_fields,
            limit=limit,
            search_params=search_params,
        )
        return res

    def hybrid_search(
        self,
        query: str,
        collection_name: str,
        dense_field: str = "vector_dense",
        output_fields: list[str] = ["text"],
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=10,
        ranker_type="weighted",
        **kwargs,
    ) -> list[dict[str, Any]]:
        filter_expression = kwargs.get("filter", "")

        semantic_results = self.semantic_search(
            collection_name=collection_name,
            query=query,
            output_fields=output_fields,
            anns_field=dense_field,
            limit=limit,
            filter=filter_expression,
        )[0]

        lexical_results = self.lexical_search(
            query=query,
            collection_name=collection_name,
            output_fields=output_fields,
            limit=limit,
            filter=filter_expression,
        )[0]

        if ranker_type.lower() == "rrf":
            ranked_results = self._rrf_merge(semantic_results, lexical_results, limit)
        else:
            ranked_results = self._weighted_merge(
                semantic_results,
                lexical_results,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                limit=limit,
            )

        return ranked_results

    def _rrf_merge(
        self,
        semantic_results: list[dict[str, Any]],
        lexical_results: list[dict[str, Any]],
        limit: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        merged_scores: dict[Any, dict[str, Any]] = {}

        for rank, result in enumerate(semantic_results, start=1):
            result_id = result.get("id", result["entity"].get("id", str(rank)))
            merged_scores[result_id] = {
                "result": result,
                "score": 1.0 / (k + rank),
            }

        for rank, result in enumerate(lexical_results, start=1):
            result_id = result.get("id", result["entity"].get("id", f"lexical-{rank}"))
            if result_id not in merged_scores:
                merged_scores[result_id] = {"result": result, "score": 0.0}
            merged_scores[result_id]["score"] += 1.0 / (k + rank)

        reranked = [
            {
                **item["result"],
                "distance": item["score"],
            }
            for item in merged_scores.values()
        ]
        reranked.sort(key=lambda item: item["distance"], reverse=True)
        return reranked[:limit]

    def _weighted_merge(
        self,
        semantic_results: list[dict[str, Any]],
        lexical_results: list[dict[str, Any]],
        dense_weight: float,
        sparse_weight: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        merged_scores: dict[Any, dict[str, Any]] = {}

        for result in semantic_results:
            result_id = result.get("id", result["entity"].get("id"))
            if result_id is None:
                continue
            merged_scores[result_id] = {
                "result": result,
                "score": dense_weight * float(result.get("distance", 0.0)),
            }

        for result in lexical_results:
            result_id = result.get("id", result["entity"].get("id"))
            if result_id is None:
                continue
            if result_id not in merged_scores:
                merged_scores[result_id] = {"result": result, "score": 0.0}
            merged_scores[result_id]["score"] += sparse_weight * float(
                result.get("distance", 0.0)
            )

        reranked = [
            {
                **item["result"],
                "distance": item["score"],
            }
            for item in merged_scores.values()
        ]
        reranked.sort(key=lambda item: item["distance"], reverse=True)
        return reranked[:limit]

    def get_dense_vector(self, query: str) -> list[float]:
        """Return a single dense embedding vector as list[float]"""
        vectors = self.embeddings.get_dense_embeddings([query])
        return vectors[0]
