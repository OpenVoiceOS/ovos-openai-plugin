# RAG memory

`ovos-openai-rag-memory-plugin` (`ovos_openai_plugin.rag_memory.PersonaServerRAGMemory`) is an
`AgentContextManager` plugin (entry point group `opm.agents.memory`). Before each turn, it searches a
vector store hosted by an OpenAI-compatible server and injects the retrieved chunks into the
conversation context that the persona's chat engine sees. The chat engine still generates the answer;
this plugin only decides what context it gets to work with, so it composes with any
`opm.agents.chat` backend rather than replacing it.

## What it talks to

The plugin sends one request per turn:

```
POST {api_url}/vector_stores/{vector_store_id}/search
{"query": "<utterance or folded history>", "max_num_results": <int>}
```

It expects a JSON body with a `data` list of `{"content": ..., "file_id": ..., "score": ...}` hits, the
shape used by the OpenAI vector stores API. It does not upload documents, create the vector store, or
run embeddings itself — the store has to already exist and be populated before this plugin is used. It
also does not call the `/files` or `/embeddings` endpoints directly; those are used ahead of time, by
whatever server or script builds the vector store (see below).

## Building the vector store

Point `api_url` at a server that implements the OpenAI files, embeddings, and vector-stores surface.
[`ovos-persona-server`](https://github.com/OpenVoiceOS/ovos-persona-server) is such a server: it exposes
`/openai/v1/files`, `/openai/v1/embeddings`, and vector-store creation/search, backed by a local
embeddings plugin, so a knowledge base can be built and queried without leaving the machine. With the
OpenAI Python SDK pointed at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="unused")

f = client.files.create(file=("cats.txt", b"cats are fluffy animals that purr."), purpose="assistants")
store = client.vector_stores.create(name="example-kb")
client.vector_stores.files.create(vector_store_id=store.id, file_id=f.id)
```

`store.id` is the `vector_store_id` this plugin is configured with. Any server that implements the same
OpenAI vector-stores contract — hosted OpenAI included — works the same way; `ovos-persona-server` is
the fully local option.

## Enabling it

Set the plugin as the persona's `memory_module`, alongside a chat engine in `solvers`:

```json
{
  "name": "kb-assistant",
  "solvers": ["ovos-chat-openai-plugin"],
  "memory_module": "ovos-openai-rag-memory-plugin",
  "ovos-openai-rag-memory-plugin": {
    "api_url": "http://localhost:8337/openai/v1",
    "vector_store_id": "vs_abc123",
    "key": null,
    "retrieval": {
      "max_num_results": 5,
      "min_score": null,
      "query_mode": "utterance",
      "query_history_turns": 3
    },
    "context": {
      "header": "Use the following context to answer the user's question.",
      "chunk_prefix": "- ",
      "chunk_separator": "\n\n",
      "include_sources": false,
      "tool_name": "search_knowledge_base"
    },
    "inject_mode": "system",
    "system_prompt": "You are a helpful assistant.",
    "max_history": 10
  }
}
```

## Configuration reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_url` | str | *(required)* | Base URL (the `/v1` root) of the OpenAI-compatible server that hosts the vector store. |
| `vector_store_id` | str | *(required)* | Id of the vector store to search, as returned by the server's vector-store creation call. |
| `key` | str | `None` | Bearer token sent as `Authorization: Bearer <key>`. Omit for local servers that do not require authentication. |
| `max_history` | int | `10` | Number of prior messages retained per session and replayed into the context on each turn. `0` disables retention. |
| `retrieval.max_num_results` | int | `5` | Maximum number of chunks requested from the search call. |
| `retrieval.min_score` | float | `None` | Drop hits scoring below this value. `None` keeps every hit the server returns. |
| `retrieval.query_mode` | str | `"utterance"` | `"utterance"` searches with the current turn only; `"history"` folds prior user turns into the query. |
| `retrieval.query_history_turns` | int | `3` | Number of prior user turns folded into the query when `query_mode` is `"history"`. |
| `context.header` | str | *(built-in instruction)* | Sentence introducing the retrieved context to the model. |
| `context.chunk_prefix` | str | `"- "` | Text prepended to each retrieved chunk. |
| `context.chunk_separator` | str | `"\n\n"` | Text joining chunks together. |
| `context.include_sources` | bool | `false` | If `true`, prefix each chunk with its `file_id` in brackets. |
| `context.tool_name` | str | `"search_knowledge_base"` | Name of the synthetic tool call used when `inject_mode` is `"tool"`. |
| `inject_mode` | str | `"system"` | How retrieved context enters the prompt; see below. |
| `system_prompt` | str | `""` | The persona's base system prompt, forwarded from the persona config. |
| `system_prompt_template` | str | `"{system}\n\n{header}\n\nContext:\n{context}"` | Template used when `inject_mode` is `"system_prompt"`. |
| `user_template` | str | `"{header}\n\nContext:\n{context}\n\nQuestion: {utterance}"` | Template used when `inject_mode` is `"user"`. |

### `inject_mode`

- `system` (default) — the persona's `system_prompt` stays its own message; retrieved context is added
  as a separate system message before the user turn. Keeps the base system prompt stable and cacheable.
- `developer` — same as `system`, but the context message uses the `developer` role instead of `system`.
- `system_prompt` — folds the context into the persona's system prompt via `system_prompt_template`,
  producing one combined system message instead of two.
- `user` — prepends the context to the final user message via `user_template`, instead of adding a
  separate message.
- `tool` — presents the context as the result of a tool call: a synthetic assistant `tool_calls` turn
  (a `search_knowledge_base` call carrying the query) followed by a `MessageRole.TOOL` message carrying
  the chunks, just before the user utterance. Requires a chat engine/backend with tool-call support.

If no hits are returned, `context` is empty and none of the above injection happens — only the base
system prompt (if any) and the user utterance are sent, unchanged.

## Local vs. hosted

Nothing in the plugin distinguishes a local server from a hosted one; both are reached the same way,
through `api_url` and the vector-stores search contract:

- **Local**: point `api_url` at a locally running `ovos-persona-server` (or any other self-hosted server
  implementing the same contract), and omit `key` if that server does not require authentication.
- **Hosted**: point `api_url` at a remote OpenAI-compatible endpoint and set `key` to the bearer token it
  expects.

## Failure modes

`_search()` is wrapped in a broad `try/except` inside `build_conversation_context`; a failed search never
raises out of the plugin. Instead it logs `RAG search failed (...); proceeding without retrieved context`
and falls back to the plain conversation — the persona still answers, just without retrieval augmentation.
This covers the two visible failure cases:

- **Endpoint missing the vector-stores/search API** — a server that only implements `/chat/completions`
  (no files/embeddings/vector-stores support) returns a non-2xx response (typically 404) from
  `resp.raise_for_status()`; retrieval is skipped for that turn.
- **Authentication failure** — a missing or wrong `key` against a server that requires one returns 401/403
  from `resp.raise_for_status()`; retrieval is skipped for that turn.

Because both failures degrade silently to a context-free answer rather than an error the user hears, the
`RAG search failed` log line is the only signal that retrieval stopped working — check it when answers
stop reflecting the knowledge base.

## Requirements

Config is only passed to memory plugins by `ovos-persona >= 0.9.0a1`; on older versions
`PersonaServerRAGMemory` will not receive its `api_url`/`vector_store_id` and will fail to construct.

---
[← Available plugins](plugins.md) · [Home](../README.md) · [Persona integration →](persona-integration.md)
