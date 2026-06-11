import numpy as np
from sklearn.neighbors import NearestNeighbors


class RetrievalBackend:
    def build_index(
    	self,
    	embeddings: np.ndarray,
    	ids: list[str],
    ):
        raise NotImplementedError

    def search(
    	self,
    	query_embeddings: np.ndarray,
    	topk: int,
    ) -> list[list[str]]:
        raise NotImplementedError


class SklearnBackend(RetrievalBackend):
    def __init__(
    	self,
    ):
        self.ids: list[str] = []
        self.index: NearestNeighbors | None = None

    def build_index(
    	self,
    	embeddings: np.ndarray,
    	ids: list[str],
    ):
        self.ids = ids
        self.index = NearestNeighbors(metric="cosine")
        self.index.fit(embeddings)

    def search(
    	self,
    	query_embeddings: np.ndarray,
    	topk: int,
    ) -> list[list[str]]:
        if self.index is None:
            raise RuntimeError("Retrieval index has not been built.")
        _, indices = self.index.kneighbors(query_embeddings, n_neighbors=topk)
        return [[self.ids[idx] for idx in row] for row in indices]


class FaissBackend(RetrievalBackend):
    def __init__(
    	self,
    ):
        try:
            import faiss
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on extra install
            raise RuntimeError(
                "faiss backend requested but faiss-cpu is not installed. "
                "Run `uv sync --extra faiss --python 3.12` first."
            ) from exc
        self.faiss = faiss
        self.ids: list[str] = []
        self.index = None

    def build_index(
    	self,
    	embeddings: np.ndarray,
    	ids: list[str],
    ):
        self.ids = ids
        normalized = embeddings.astype("float32")
        self.faiss.normalize_L2(normalized)
        self.index = self.faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)

    def search(
    	self,
    	query_embeddings: np.ndarray,
    	topk: int,
    ) -> list[list[str]]:
        if self.index is None:
            raise RuntimeError("Retrieval index has not been built.")
        normalized = query_embeddings.astype("float32")
        self.faiss.normalize_L2(normalized)
        _, indices = self.index.search(normalized, topk)
        return [[self.ids[idx] for idx in row] for row in indices]


def build_backend(
	name: str,
) -> RetrievalBackend:
    if name == "sklearn":
        return SklearnBackend()
    if name == "faiss":
        return FaissBackend()
    raise ValueError(f"Unsupported retrieval backend: {name}")
