"""The OpenAI-compatible text embedder, against a local server that speaks the /v1/embeddings shape."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from ovos_plugin_manager.embeddings import find_text_embeddings_plugins

from ovos_openai_plugin.embeddings import OpenAIEmbeddings


def vector(text):
    """The server's embedding of ``text``: computed here, independently of the plugin."""
    return [float(len(text)), float(text.count("a")), 1.0]


@pytest.fixture
def server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            if any(text == "too long" for text in inputs):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "input too long"}}')
                return
            # the API gives each item an index; send them reversed so the client must use it
            data = [{"object": "embedding", "index": i, "embedding": vector(t)} for i, t in enumerate(inputs)][::-1]
            reply = json.dumps({"object": "list", "data": data, "model": body["model"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", calls
    httpd.shutdown()


TEXTS = ["banana", "a", "casa amarela", "rio"]


def test_a_batch_is_one_request_with_the_rows_in_input_order(server):
    url, calls = server
    rows = OpenAIEmbeddings({"api_url": url, "model": "m", "key": "k"}).get_embeddings_batch(TEXTS)
    assert [list(r) for r in rows] == [vector(t) for t in TEXTS]
    assert len(calls) == 1
    assert calls[0]["path"] == "/v1/embeddings"
    assert calls[0]["input"] == TEXTS
    assert calls[0]["model"] == "m"
    assert calls[0]["auth"] == "Bearer k"


def test_a_batch_larger_than_batch_size_is_split_in_order(server):
    url, calls = server
    rows = OpenAIEmbeddings({"api_url": url, "model": "m", "batch_size": 3}).get_embeddings_batch(TEXTS * 2)
    assert [list(r) for r in rows] == [vector(t) for t in TEXTS * 2]
    assert [len(c["input"]) for c in calls] == [3, 3, 2]
    assert calls[0]["auth"] is None


def test_one_text_is_one_row(server):
    url, _ = server
    assert list(OpenAIEmbeddings({"api_url": url, "model": "m"}).get_embeddings("casa")) == vector("casa")


def test_dimensions_is_sent_only_when_set(server):
    url, calls = server
    OpenAIEmbeddings({"api_url": url, "model": "m"}).get_embeddings("a")
    OpenAIEmbeddings({"api_url": url, "model": "m", "dimensions": 2}).get_embeddings("a")
    assert "dimensions" not in calls[0]
    assert calls[1]["dimensions"] == 2


def test_an_empty_batch_sends_nothing(server):
    url, calls = server
    assert OpenAIEmbeddings({"api_url": url, "model": "m"}).get_embeddings_batch([]) == []
    assert calls == []


def test_a_refused_request_raises_with_the_server_message(server):
    url, _ = server
    with pytest.raises(RuntimeError, match="400.*input too long"):
        OpenAIEmbeddings({"api_url": url, "model": "m"}).get_embeddings_batch(["a", "too long"])


def test_a_model_is_required():
    with pytest.raises(ValueError, match="model"):
        OpenAIEmbeddings({"api_url": "http://127.0.0.1:1/v1"})


def test_the_rows_are_numpy_vectors(server):
    url, _ = server
    (row,) = OpenAIEmbeddings({"api_url": url, "model": "m"}).get_embeddings_batch(["a"])
    assert isinstance(row, np.ndarray) and row.dtype == np.float32


def test_the_plugin_is_registered_as_a_text_embedder():
    assert find_text_embeddings_plugins().get("ovos-openai-embeddings-plugin") is OpenAIEmbeddings
