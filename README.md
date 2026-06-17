# <img src='https://raw.githack.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> OVOS OpenAI Plugin

This plugin is designed to leverage the **OpenAI API** for various functionalities within the OpenVoiceOS ecosystem. It provides a set of OVOS plugins that interact with OpenAI's services. Crucially, it is also compatible with **self-hosted OpenAI-compatible alternatives**, such as the [OVOS Persona Server](https://github.com/OpenVoiceOS/ovos-persona-server), or any other project that implements the full suite of OpenAI API endpoints (Chat Completions, Embeddings, Files, and Vector Stores). This flexibility allows you to choose between cloud-based OpenAI services or a local, private setup.

Specifically, this plugin provides:

  - `ovos-solver-openai-plugin` for general chat completions, primarily for usage with [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) (and in older ovos releases with [ovos-skill-fallback-chatgpt](https://www.google.com/search?q=))
  - `ovos-openai-rag-memory-plugin` for Retrieval Augmented Generation (as a persona memory plugin) using a compatible backend (like `ovos-persona-server`) as a knowledge source.
  - `ovos-dialog-transformer-openai-plugin` to rewrite OVOS dialogs just before TTS executes in [ovos-audio](https://github.com/OpenVoiceOS/ovos-audio)
  - `ovos-summarizer-openai-plugin` to summarize text, not used directly but provided for consumption by other plugins/skills

-----

## Install

`pip install ovos-openai-plugin`

-----

## Persona Usage

To create your own persona using a OpenAI compatible server create a .json in `~/.config/ovos_persona/llm.json`:

```json
{
  "name": "My Local LLM",
  "solvers": [
    "ovos-solver-openai-plugin"
  ],
  "ovos-solver-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "sk-xxxx",
    "system_prompt": "You are helping assistant who gives very short and factual answers in maximum twenty words and you don't use emojis"
  }
}
```

Then say "Chat with {name_from_json}" to enable it, more details can be found in [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) README

This plugins also provides a default "Remote LLama" demo persona, it points to a public server hosted by @goldyfruit.

-----

## RAG Memory Plugin

`ovos-openai-rag-memory-plugin` (`PersonaServerRAGMemory`) enables **Retrieval
Augmented Generation (RAG)** as a persona **memory plugin** rather than a solver. It
hooks the persona's context-building step: before each turn it searches a vector
store hosted by a compatible backend (e.g. [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server))
and injects the retrieved chunks into the conversation context. Your persona's normal
chat engine then answers — so RAG composes with **any** chat backend instead of owning
the chat round-trip.

This is useful for grounding answers in your own documentation / notes / data and
reducing hallucinations.

### How it works

1. **Search** — `build_conversation_context` sends the query to the backend's vector
   store search endpoint.
2. **Retrieve** — the backend returns relevant text chunks.
3. **Inject** — the chunks are added to the context per `inject_mode` (a separate
   system message by default).
4. **Generate** — the persona's chat engine answers with the augmented context.

### Configuration

Set it as the persona's `memory_module` in `~/.config/ovos_persona/<persona>.json`.
You need a compatible backend running with a populated vector store and its
`vector_store_id`. Requires `ovos-persona` with memory-plugin config passing.

```json
{
  "name": "My RAG Assistant",
  "solvers": ["ovos-solver-openai-plugin"],
  "memory_module": "ovos-openai-rag-memory-plugin",
  "ovos-openai-rag-memory-plugin": {
    "api_url": "http://localhost:8337/openai/v1",
    "vector_store_id": "vs_your_vector_store_id_here",
    "key": "sk-xxxx",
    "system_prompt": "You are a helpful assistant.",
    "inject_mode": "system",
    "retrieval": { "max_num_results": 5, "min_score": null, "query_mode": "utterance" },
    "context": { "include_sources": false },
    "max_history": 10
  }
}
```

**Strategies** (all configurable):

- `inject_mode` — `system` (separate system message, default), `system_prompt` (fold
  into the persona's system prompt), `developer` (developer-role message), or `user`
  (prepend to the user turn).
- `retrieval.query_mode` — `utterance` (default) or `history` (fold recent turns into
  the query); `max_num_results`, `min_score`.
- `context` — `include_sources`, `chunk_prefix`, `chunk_separator`, `header`.

-----

## Dialog Transformer

You can rewrite text dynamically based on specific personas, such as simplifying explanations or mimicking a specific tone.

#### Example Usage:

  - **`rewrite_prompt`:** `"rewrite the text as if you were explaining it to a 5-year-old"`
  - **Input:** `"Quantum mechanics is a branch of physics that describes the behavior of particles at the smallest scales."`
  - **Output:** `"Quantum mechanics is like a special kind of science that helps us understand really tiny things."`

Examples of `rewrite_prompt` Values:

  - `"rewrite the text as if it was an angry old man speaking"`
  - `"Add more 'dude'ness to it"`
  - `"Explain it like you're teaching a child"`

To enable this plugin, add the following to your `mycroft.conf`:

```json
"dialog_transformers": {
    "ovos-dialog-transformer-openai-plugin": {
        "system_prompt": "Your task is to rewrite text as if it was spoken by a different character",
        "rewrite_prompt": "rewrite the text as if you were explaining it to a 5-year-old"
    }
}
```

> 💡 the user utterance will be appended after `rewrite_prompt` for the actual query

-----

## Direct Usage

```python
from ovos_solver_openai_persona import OpenAIPersonaSolver

bot = OpenAIPersonaSolver({"key": "sk-XXX",
                           "persona": "helpful, creative, clever, and very friendly"})
print(bot.get_spoken_answer("describe quantum mechanics in simple terms"))
# Quantum mechanics is a branch of physics that deals with the behavior of particles on a very small scale, such as atoms and subatomic particles. It explores the idea that particles can exist in multiple states at once and that their behavior is not predictable in the traditional sense.
print(bot.spoken_answer("Quem encontrou o caminho maritimo para o Brazil", lang="pt-pt"))
# Explorador português Pedro Álvares Cabral é creditado com a descoberta do Brasil em 1500

```

-----

## Remote Persona / Proxies

You can run any persona behind an **OpenAI-compatible server** (such as [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server)).

This allows you to offload the workload to a standalone server, either for performance reasons or to keep API keys in a single safe place. Then, you just configure this plugin to point to your self-hosted server as if it were the official OpenAI API.
