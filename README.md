# <img src='https://raw.githack.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> OVOS OpenAI Plugin

This package uses the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) to
provide OpenAI-compatible plugins for OpenVoiceOS. It works with any server that exposes the OpenAI
`/chat/completions` contract, such as OpenAI, [ollama](https://ollama.com), llama.cpp, vLLM, or LocalAI.
Point `api_url` at the server's `/v1` base to use it.

This package provides:

| Plugin | Entry point | Type | Purpose |
|--------|-------------|------|---------|
| `ovos-chat-openai-plugin` | `opm.agents.chat` | `ChatEngine` | Multi-turn chat agent (native tool/function calling) for [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) |
| `ovos-openai-rag-memory-plugin` | `opm.agents.memory` | `AgentContextManager` | RAG memory: inject vector-store context from an [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server) |
| `ovos-summarizer-openai-plugin` | `opm.agents.summarizer` | `SummarizerEngine` | Summarize text for other plugins/skills |
| `ovos-translate-openai-plugin` | `opm.lang.translate` | `LanguageTranslator` | LLM-backed text translation |
| `ovos-lang-detect-openai-plugin` | `opm.lang.detect` | `LanguageDetector` | LLM-backed language detection |
| `ovos-dialog-transformer-openai-plugin` | `opm.transformer.dialog` | `DialogTransformer` | Rewrite OVOS dialogs just before TTS in [ovos-audio](https://github.com/OpenVoiceOS/ovos-audio) |

> **Breaking change:** solver plugins are [deprecated in ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager/pull/365).
> This release migrates from the legacy `QuestionSolver`/`ChatMessageSolver` to the new
> [agents framework](https://github.com/OpenVoiceOS/ovos-plugin-manager) (`AbstractAgentEngine`).
> The old `ovos-solver-openai-plugin` entry point and the `OpenAIChatCompletionsSolver` and
> `OpenAIPersonaSolver` classes have been removed. Personas must now reference `ovos-chat-openai-plugin`.
> This release requires `ovos-plugin-manager>=2.2.3a1` and `ovos-persona>=0.9.0a1`.

## Install

```bash
pip install ovos-openai-plugin
```

## Persona Usage

To create your own persona using an OpenAI-compatible server, create a `.json` in `~/.config/ovos_persona/llm.json`:

```json
{
  "name": "My Local LLM",
  "solvers": [
    "ovos-chat-openai-plugin"
  ],
  "ovos-chat-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "sk-xxxx",
    "system_prompt": "You are a helpful assistant who gives very short and factual answers in maximum twenty words and you don't use emojis",
    "model": "llama3.1:8b"
  }
}
```

> The `solvers` key name is kept for backwards compatibility with persona JSON files. It now accepts
> agent plugin names (chat engines) in addition to legacy solvers.

Say "Chat with {name_from_json}" to enable it. See the
[ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) README for more details.

This plugin also provides a default "Remote LLama" demo persona, pointing to a public server hosted by @goldyfruit.

## RAG memory

`ovos-openai-rag-memory-plugin` (`PersonaServerRAGMemory`) is a persona memory
plugin. Before each turn, it searches a vector store on an
[ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server) and injects
the retrieved chunks into the conversation context. The persona's chat engine then
answers. This plugin composes with any chat backend instead of owning the chat round-trip.

Set it as the persona's `memory_module`:

```json
{
  "name": "kb-assistant",
  "solvers": ["ovos-chat-openai-plugin"],
  "memory_module": "ovos-openai-rag-memory-plugin",
  "ovos-openai-rag-memory-plugin": {
    "api_url": "http://localhost:8337/openai/v1",
    "vector_store_id": "vs_...",
    "inject_mode": "system",
    "retrieval": {"max_num_results": 5}
  }
}
```

`inject_mode` selects how retrieved context enters the prompt: `system` (separate
system message, default), `system_prompt`, `developer`, `user`, or `tool` (a
synthetic `search_knowledge_base` tool-call result). Retrieval (`max_num_results`,
`min_score`, `query_mode`) and context formatting are configurable. This feature
requires an `ovos-persona` version that passes config to memory plugins. See
[docs/rag-memory.md](docs/rag-memory.md) for the full configuration reference, how
to build the vector store this plugin searches (files + embeddings, local or
hosted), and its failure modes.

## Dialog Transformer

Rewrite text dynamically based on a persona, such as simplifying explanations or mimicking a tone.

**Example:**
- **`rewrite_prompt`:** `"rewrite the text as if you were explaining it to a 5-year-old"`
- **Input:** `"Quantum mechanics is a branch of physics that describes the behavior of particles at the smallest scales."`
- **Output:** `"Quantum mechanics is like a special kind of science that helps us understand really tiny things."`

To enable this plugin, add the following to your `mycroft.conf`:

```json
"dialog_transformers": {
    "ovos-dialog-transformer-openai-plugin": {
        "api_url": "https://api.openai.com/v1",
        "key": "sk-xxxx",
        "model": "gpt-4o-mini",
        "system_prompt": "Your task is to rewrite text as if it was spoken by a different character",
        "rewrite_prompt": "rewrite the text as if you were explaining it to a 5-year-old"
    }
}
```

> The actual dialog text is appended after `rewrite_prompt` in the request.

## Direct Usage

```python
from ovos_openai_plugin import OpenAIChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

bot = OpenAIChatEngine({
    "key": "sk-XXX",
    "model": "gpt-4o-mini",
    "system_prompt": "You are helpful, creative, clever, and very friendly",
})

# one-shot answer
print(bot.get_response("describe quantum mechanics in simple terms"))

# multi-turn chat
reply = bot.continue_chat([
    AgentMessage(MessageRole.USER, "what is the capital of France?")
])
print(reply.content)

# stream complete sentences (TTS friendly)
for sentence in bot.stream_sentences([AgentMessage(MessageRole.USER, "tell me a short story")]):
    print(sentence)
```

Translation and summarization:

```python
from ovos_openai_plugin import OpenAITextTranslator, OpenAISummarizer

tx = OpenAITextTranslator({"key": "sk-XXX", "model": "gpt-4o-mini"})
print(tx.translate("hello world", target="es-es"))

summary = OpenAISummarizer({"key": "sk-XXX", "model": "gpt-4o-mini"})
print(summary.summarize("a very long document ..."))
```

## Remote Persona / Proxies

You can run any persona behind an OpenAI-compatible server with
[ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server). This moves the workload to a
standalone server, either for performance or to keep API keys in a single safe place. Then configure this
plugin to point at your persona server as if it were OpenAI.

## Documentation

- [Configuration reference](docs/configuration.md)
- [Available plugins](docs/plugins.md)
- [Persona integration](docs/persona-integration.md)
- [RAG memory](docs/rag-memory.md)

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
