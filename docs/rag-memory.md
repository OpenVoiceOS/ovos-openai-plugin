# RAG memory

`ovos-openai-rag-memory-plugin` (`ovos_openai_plugin.rag_memory.PersonaServerRAGMemory`) is an
`AgentContextManager` plugin (entry point group `opm.agents.memory`). Before each turn, it searches a
vector store hosted by an OpenAI-compatible server and injects the retrieved chunks into the
conversation context that the persona's chat engine sees. The chat engine still generates the answer.
This plugin only decides what context the engine gets, so it composes with any `opm.agents.chat`
backend and does not replace it.

## What it talks to

The plugin sends one request per turn:

```
POST {api_url}/vector_stores/{vector_store_id}/search
{"query": "<utterance or folded history>", "max_num_results": <int>}
```

It expects a JSON body with a `data` list of `{"content": ..., "file_id": ..., "score": ...}` hits, the
shape used by the OpenAI vector stores API. It does not upload documents, create the vector store, or
run embeddings itself. The store must already exist and hold documents before you use this plugin. It
also does not call the `/files` or `/embeddings` endpoints directly. The server or script that builds the
vector store uses those endpoints ahead of time (see below).

## Building the vector store

Point `api_url` at a server that implements the OpenAI files, embeddings, and vector-stores surface.
[`ovos-persona-server`](https://github.com/OpenVoiceOS/ovos-persona-server) is such a server. It exposes
`/openai/v1/files`, `/openai/v1/embeddings`, and vector-store creation and search, backed by a local
embeddings plugin. You can build and query a knowledge base without leaving the machine. Use the
OpenAI Python SDK, pointed at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="unused")

f = client.files.create(file=("cats.txt", b"cats are fluffy animals that purr."), purpose="assistants")
store = client.vector_stores.create(name="example-kb")
client.vector_stores.files.create(vector_store_id=store.id, file_id=f.id)
```

`store.id` is the `vector_store_id` this plugin is configured with. Any server that implements the same
OpenAI vector-stores contract works the same way, and that includes hosted OpenAI. `ovos-persona-server`
is the fully local option.

## Enabling it

Set the plugin as the persona's `memory_module`, alongside a chat engine in `solvers`:

```json
{
  "name": "kb-assistant",
  "solvers": ["ovos-chat-openai-plugin"],
  "ovos-chat-openai-plugin": {
    "api_url": "http://localhost:8080/v1",
    "key": "sk-none",
    "model": "my-model",
    "allow_system_prompts": true
  },
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

### With `ovos-chat-openai-plugin`

`ovos-chat-openai-plugin` removes every system message unless
`allow_system_prompts` is `true`. The default `inject_mode` is `system`, so the
retrieved context is removed with it. With the configuration above and
`allow_system_prompts` off, the plugin's `system_prompt` and the retrieved context are two system messages, and
the log says `dropped 2 system message(s) before the request`.

Set `allow_system_prompts` to `true`, or set `inject_mode` to `user`. With
`inject_mode` set to `user` and `allow_system_prompts` off, the plugin's own
`system_prompt` is still removed, and the log says `dropped 1 system message(s)`.

A run in which the vector store search returned one chunk about a bakery gave these
answers to "When does the Pixel Bakery close on Saturday?":

| `allow_system_prompts` | `inject_mode` | Answer |
|---|---|---|
| `false` | `system` | I don't have any information about a specific "Pixel Bakery" or its operating hours. Can you please provide more context or clarify which Pixel Bakery you are referring to? I'll do my best to help. |
| `true` | `system` | The Pixel Bakery closes at 14:30 on Saturdays. |
| `false` | `user` | The Pixel Bakery closes at 14:30 on Saturdays. |

## Configuration reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_url` | str | *(required)* | Base URL (the `/v1` root) of the OpenAI-compatible server that hosts the vector store. |
| `vector_store_id` | str | *(required)* | Id of the vector store to search, as returned by the server's vector-store creation call. |
| `key` | str | `None` | Bearer token sent as `Authorization: Bearer <key>`. Omit for local servers that do not require authentication. |
| `max_history` | int | `10` | Number of prior messages retained per session and replayed into the context on each turn. `0` disables retention. |
| `retrieval.max_num_results` | int | `5` | Maximum number of chunks requested from the search call. |
| `retrieval.min_score` | float | `None` | Drop hits scoring below this value. `None` keeps every hit the server returns. |
| `retrieval.query_mode` | str | `"utterance"` | `"utterance"` searches with the current turn only. `"history"` folds prior user turns into the query. |
| `retrieval.query_history_turns` | int | `3` | Number of prior user turns folded into the query when `query_mode` is `"history"`. |
| `context.header` | str | *(built-in instruction)* | Sentence introducing the retrieved context to the model. |
| `context.chunk_prefix` | str | `"- "` | Text prepended to each retrieved chunk. |
| `context.chunk_separator` | str | `"\n\n"` | Text joining chunks together. |
| `context.include_sources` | bool | `false` | If `true`, prefix each chunk with its `file_id` in brackets. |
| `context.tool_name` | str | `"search_knowledge_base"` | Name of the synthetic tool call used when `inject_mode` is `"tool"`. |
| `inject_mode` | str | `"system"` | How retrieved context enters the prompt. See below. |
| `system_prompt` | str | `""` | The persona's base system prompt, forwarded from the persona config. |
| `system_prompt_template` | str | `"{system}\n\n{header}\n\nContext:\n{context}"` | Template used when `inject_mode` is `"system_prompt"`. |
| `user_template` | str | `"{header}\n\nContext:\n{context}\n\nQuestion: {utterance}"` | Template used when `inject_mode` is `"user"`. |

### `inject_mode`

- `system` (default): the persona's `system_prompt` stays its own message. The plugin adds the retrieved
  context as a separate system message before the user turn. This keeps the base system prompt stable
  and cacheable.
- `developer`: the same as `system`, but the context message uses the `developer` role.
- `system_prompt`: the plugin folds the context into the persona's system prompt with
  `system_prompt_template`. The result is one combined system message instead of two.
- `user`: the plugin prepends the context to the final user message with `user_template`. It adds no
  separate message.
- `tool`: the plugin presents the context as the result of a tool call. A synthetic assistant
  `tool_calls` turn (a `search_knowledge_base` call that carries the query) comes first. A
  `MessageRole.TOOL` message that carries the chunks follows, just before the user utterance. This mode
  needs a chat engine with tool-call support.

If the search returns no hits, `context` is empty and no injection happens. The plugin sends only the
base system prompt (if any) and the user utterance, unchanged.

## Local vs. hosted

The plugin does not distinguish a local server from a hosted one. It reaches both the same way, through
`api_url` and the vector-stores search contract:

- **Local**: point `api_url` at a locally running `ovos-persona-server` (or any other self-hosted server
  implementing the same contract), and omit `key` if that server does not require authentication.
- **Hosted**: point `api_url` at a remote OpenAI-compatible endpoint and set `key` to the bearer token it
  expects.

## Failure modes

`build_conversation_context` wraps `_search()` in a broad `try/except`, so a failed search never raises
out of the plugin. The plugin logs `RAG search failed (...); proceeding without retrieved context` and
falls back to the plain conversation. The persona still answers, but without retrieval augmentation.
This covers the two visible failure cases:

- **Endpoint missing the vector-stores search API.** A server that only implements `/chat/completions`
  (no files, embeddings or vector-stores support) returns a non-2xx response, typically 404, from
  `resp.raise_for_status()`. The plugin skips retrieval for that turn.
- **Authentication failure.** A missing or wrong `key` against a server that requires one returns 401 or
  403 from `resp.raise_for_status()`. The plugin skips retrieval for that turn.

Both failures degrade silently to a context-free answer. The user hears no error. The
`RAG search failed` log line is the only signal that retrieval stopped. Check it when answers stop
reflecting the knowledge base.

---
[← Available plugins](plugins.md) · [Home](../README.md) · [Persona integration →](persona-integration.md)
