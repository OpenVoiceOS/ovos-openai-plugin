# Text embeddings

`ovos-openai-embeddings-plugin` turns text into vectors through any server with an OpenAI-compatible `/v1/embeddings` endpoint: OpenAI, Ollama, llama.cpp `llama-server --embeddings`, vLLM, LiteLLM or ovos-persona-server. Retrieval, deduplication and the RAG vector stores use these vectors.

- **Entry point group:** `opm.embeddings.text`
- **Class:** `ovos_openai_plugin.embeddings.OpenAIEmbeddings` (`TextEmbedder`)

## Use it

```python
from ovos_plugin_manager.embeddings import load_text_embeddings_plugin

embedder = load_text_embeddings_plugin("ovos-openai-embeddings-plugin")({
    "api_url": "http://localhost:11434/v1",
    "model": "nomic-embed-text",
})
vectors = embedder.get_embeddings_batch(["Qual é a capital de Portugal?", "Lisboa é a capital."])
```

`get_embeddings(text)` returns one vector. `get_embeddings_batch(texts)` returns one vector per text, in the order of `texts`. It sends one request for each `batch_size` texts. Use the batch method for many texts, because one request per text is slow over a network.

Each vector is a `numpy` `float32` array, as the server returns it. The plugin does not normalise the vectors. Most embedding models return vectors of length 1, but check your model before you use a dot product as a cosine similarity.

## Configuration

| Key | Type | Default | Meaning |
|---|---|---|---|
| `api_url` | `str` | `https://api.openai.com/v1` | The `/v1` root of the server. `/embeddings` is appended. |
| `model` | `str` | required | The embedding model to ask for. |
| `key` | `str` | unset | Bearer token, for a server that needs one. |
| `dimensions` | `int` | unset | Vector size to ask for. Only some models, such as `text-embedding-3-small`, can shorten their output. |
| `batch_size` | `int` | `128` | Texts sent in one request. Lower it when the server limits the input size of a request. |
| `timeout` | `float` | `30` | Seconds to wait for each reply. |

## Failures

A reply that is not a 2xx status raises `RuntimeError` with the status and the start of the reply body. A reply with a different number of vectors than texts also raises `RuntimeError`. The plugin sorts the vectors by the `index` of each item, so a server that returns them out of order still gives the right vector for each text.

---
[← Available plugins](plugins.md) · [Home](../README.md) · [Persona integration →](persona-integration.md)
