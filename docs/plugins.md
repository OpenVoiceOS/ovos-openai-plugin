# Available plugins

This package ships six plugins. Five are backed by a shared OpenAI Chat Completions client
(`ovos_openai_plugin.api.OpenAIChatCompletions`); the RAG memory plugin instead talks to an
OpenAI-compatible vector-stores search API.

## `ovos-chat-openai-plugin`: Chat engine

- **Entry point group:** `opm.agents.chat`
- **Class:** `ovos_openai_plugin.chat.OpenAIChatEngine` (`ChatEngine`)

A multi-turn conversational agent for ovos-persona. Implements:

- `continue_chat(messages, ...) -> AgentMessage`: full response
- `stream_tokens(messages, ...)`: raw token stream (not TTS-safe)
- `stream_sentences(messages, ...)`: complete sentences, buffered via
  [`sentence-stream`](https://pypi.org/project/sentence-stream/) (TTS-safe)
- `get_response(utterance, ...)`: convenience one-shot helper (from the base class)

System prompt handling is controlled by `system_prompt` and `allow_system_prompts`
(see [configuration](configuration.md)).

### Tool / function calling

`OpenAIChatEngine` sets `supports_tools = True`. Pass `ToolBox` object(s) (or OpenAI
tool dicts) to `continue_chat(..., tools=...)`. The engine exposes them to the model.
When the model requests a tool, it returns an assistant `AgentMessage` whose
`tool_calls` (a list of `ToolCall`) are populated. Feed the results back as
`MessageRole.TOOL` messages and call again to continue. The
[`ovos-native-toolcall-loop`](https://github.com/OpenVoiceOS/ovos-agentic-loop)
engine drives this loop for you. This plugin only needs to be the `brain`.

## `ovos-openai-rag-memory-plugin`: RAG memory

- **Entry point group:** `opm.agents.memory`
- **Class:** `ovos_openai_plugin.rag_memory.PersonaServerRAGMemory` (`AgentContextManager`)

A persona memory plugin: before each turn, searches a vector store on an OpenAI-compatible server and
injects the retrieved chunks into the conversation context that the chat engine sees. See
[RAG memory](rag-memory.md) for the full configuration reference, how to build the vector store it
searches, and its failure modes.

## `ovos-summarizer-openai-plugin`: Summarizer

- **Entry point group:** `opm.agents.summarizer`
- **Class:** `ovos_openai_plugin.summarizer.OpenAISummarizer` (`SummarizerEngine`)

Summarizes a document via `summarize(document, lang=None) -> str`. The prompt is built from
`prompt_template` (must contain `{content}`).

## `ovos-translate-openai-plugin`: Translator

- **Entry point group:** `opm.lang.translate`
- **Class:** `ovos_openai_plugin.translate.OpenAITextTranslator` (`LanguageTranslator`)

`translate(text, target=None, source=None) -> str`. Language codes are expanded to human-readable
names (via `langcodes`) before being inserted into the prompt. If `target` is omitted, the configured
`lang` from OVOS configuration is used. If `source` is omitted, the model auto-detects it.

## `ovos-lang-detect-openai-plugin`: Language detector

- **Entry point group:** `opm.lang.detect`
- **Class:** `ovos_openai_plugin.translate.OpenAITextLangDetector` (`LanguageDetector`)

`detect(text) -> str` returns a standardized BCP-47 tag. `detect_probs(text) -> Dict[str, float]`
returns `{lang: 1.0}`.

## `ovos-dialog-transformer-openai-plugin`: Dialog transformer

- **Entry point group:** `opm.transformer.dialog`
- **Class:** `ovos_openai_plugin.dialog_transformers.OpenAIDialogTransformer` (`DialogTransformer`)

Rewrites dialog text just before TTS. `transform(dialog, context) -> (text, context)`. The rewrite
prompt comes from `context["prompt"]` or the configured `rewrite_prompt`. If neither is set, the dialog
is returned unchanged.

## Demo persona

The `opm.plugin.persona` entry point `Remote Llama` exposes a ready-made persona
(`ovos_openai_plugin.LLAMA_DEMO`) pointed at a public OpenAI-compatible LLama server.

---
[← Configuration reference](configuration.md) · [Home](../README.md) · [Persona integration →](persona-integration.md)
