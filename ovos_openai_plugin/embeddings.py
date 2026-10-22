from typing import Any, Dict, List, Optional

import numpy as np
import requests
from ovos_plugin_manager.templates.embeddings import EmbeddingsArray, TextEmbedder


class OpenAIEmbeddings(TextEmbedder):
    """Text embeddings from any server with an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Config:

    ``api_url`` : the ``/v1`` root (default ``https://api.openai.com/v1``); ``/embeddings`` is appended.
    ``model`` : the embedding model to ask for. Required.
    ``key`` : bearer token, when the server needs one.
    ``dimensions`` : vector size to ask for, for models that can shorten their output.
    ``batch_size`` : texts sent in one request (default 128).
    ``timeout`` : seconds to wait for each reply (default 30).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = self.config.get("model")
        if not self.model:
            raise ValueError("OpenAIEmbeddings needs a 'model' in its config")
        self.url = (self.config.get("api_url") or "https://api.openai.com/v1").rstrip("/") + "/embeddings"
        self.key = self.config.get("key") or ""
        self.dimensions = self.config.get("dimensions")
        self.batch_size = int(self.config.get("batch_size", 128))
        self.timeout = float(self.config.get("timeout", 30))
        self.session = requests.Session()

    def get_embeddings(self, text: str) -> EmbeddingsArray:
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[EmbeddingsArray]:
        """One request for each ``batch_size`` texts; the rows come back in the order of ``texts``."""
        rows: List[EmbeddingsArray] = []
        for start in range(0, len(texts), self.batch_size):
            rows += self._embed(texts[start:start + self.batch_size])
        return rows

    def _embed(self, texts: List[str]) -> List[EmbeddingsArray]:
        body: Dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        resp = self.session.post(self.url, json=body, headers=headers, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"{self.url}: {resp.status_code} {resp.text[:300]}")
        data = sorted(resp.json()["data"], key=lambda item: item["index"])
        if len(data) != len(texts):
            raise RuntimeError(f"{self.url}: {len(data)} embeddings for {len(texts)} texts")
        return [np.asarray(item["embedding"], dtype=np.float32) for item in data]
