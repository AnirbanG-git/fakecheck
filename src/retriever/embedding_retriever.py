# src/retriever/embedding_retriever.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import os, json
import numpy as np
import torch
from torch import amp
from sentence_transformers import SentenceTransformer
import faiss

@dataclass
class Doc:
    doc_id: str
    text: str
    meta: Dict

class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 128,
        use_gpu_faiss: bool = True,
        use_amp: bool = True,
        normalize_embeddings: bool = True,
        # tolerated for nodes/CLI
        persist_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        **kwargs,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.use_gpu_faiss = (device == "cuda") and use_gpu_faiss
        self.use_amp = (device == "cuda") and use_amp
        self.normalize = normalize_embeddings

        # optional dirs
        self.persist_dir = persist_dir
        self.index_dir = index_dir or persist_dir

        # encoder
        self.model = SentenceTransformer(model_name, device=self.device)

        # in-memory store
        self.index: Optional[faiss.Index] = None
        self.texts: List[str] = []
        self.meta: List[Dict] = []
        self._dim: Optional[int] = None

    # ---------- encoding ----------
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 384), dtype="float32")
        vecs = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i:i + self.batch_size]
                if self.use_amp:
                    with amp.autocast(device_type="cuda"):
                        e = self.model.encode(
                            chunk,
                            convert_to_numpy=True,
                            normalize_embeddings=False,
                            show_progress_bar=False,
                            batch_size=self.batch_size,
                        )
                else:
                    e = self.model.encode(
                        chunk,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        show_progress_bar=False,
                        batch_size=self.batch_size,
                    )
                vecs.append(e.astype("float32"))
        X = np.vstack(vecs) if vecs else np.zeros((0, self._dim or 384), dtype="float32")
        if self.normalize and len(X):
            faiss.normalize_L2(X)
        return X

    # ---------- build / load ----------
    def build(self, docs: List[Doc]) -> None:
        self.texts = [d.text for d in docs]
        self.meta  = [d.meta for d in docs]
        X = self._encode_texts(self.texts)
        if X.size == 0:
            raise ValueError("No documents to build index on.")
        self._dim = int(X.shape[1])

        cpu_index = faiss.IndexFlatIP(self._dim)
        if self.use_gpu_faiss:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.add(X)
            self.index = gpu_index
        else:
            cpu_index.add(X)
            self.index = cpu_index

    def build_or_load(self, docs: List[Doc]) -> None:
        """
        If a persisted index exists (index.faiss + texts.npy + meta.json), load it.
        Otherwise build from docs and save if persist_dir is set.
        """
        if self.persist_dir and self._persist_exists(self.persist_dir):
            self.load(self.persist_dir)
            return
        # else build new
        self.build(docs)
        if self.persist_dir:
            self.save(self.persist_dir)

    # ---------- persistence ----------
    def _persist_exists(self, d: str) -> bool:
        return (
            os.path.exists(os.path.join(d, "index.faiss")) and
            os.path.exists(os.path.join(d, "texts.npy")) and
            os.path.exists(os.path.join(d, "meta.json"))
        )

    def save(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        # ensure CPU index for saving
        idx = self.index
        if idx is None:
            raise RuntimeError("No index to save.")
        if self.use_gpu_faiss and not isinstance(idx, faiss.IndexFlat):
            idx = faiss.index_gpu_to_cpu(idx)
        faiss.write_index(idx, os.path.join(d, "index.faiss"))
        np.save(os.path.join(d, "texts.npy"), np.array(self.texts, dtype=object))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(self.meta, f)

    def load(self, d: str) -> None:
        # read CPU index then move to GPU if needed
        idx_path = os.path.join(d, "index.faiss")
        cpu_index = faiss.read_index(idx_path)
        if self.use_gpu_faiss:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index
        self.texts = np.load(os.path.join(d, "texts.npy"), allow_pickle=True).tolist()
        with open(os.path.join(d, "meta.json"), "r") as f:
            self.meta = json.load(f)
        # infer dim
        if self.index is None:
            raise RuntimeError("Failed to load FAISS index.")
        self._dim = self.index.d

    # ---------- querying ----------
    def query(self, q: str, top_k: int = 10):
        if self.index is None:
            raise RuntimeError("Index not built/loaded. Call build_or_load() or build() first.")
        qv = self._encode_texts([q])
        D, I = self.index.search(qv, top_k)
        hits = []
        for j, i in enumerate(I[0]):
            i = int(i)
            hits.append((self.texts[i], self.meta[i], float(D[0][j])))
        return hits
