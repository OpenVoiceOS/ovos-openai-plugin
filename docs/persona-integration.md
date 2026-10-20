# Persona integration

`ovos-chat-openai-plugin` plugs into [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) as a
chat agent. Personas are JSON files placed in `~/.config/ovos_persona/`.

## Minimal persona

```json
{
  "name": "My Local LLM",
  "solvers": [
    "ovos-chat-openai-plugin"
  ],
  "ovos-chat-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "sk-xxxx",
    "model": "llama3.1:8b",
    "system_prompt": "You are a helpful assistant who gives short, factual answers."
  }
}
```

- The `solvers` list contains the **plugin name** of the chat engine that answers.
- A top-level key matching the plugin name holds that plugin's configuration.

Enable it at runtime by saying *"Chat with My Local LLM"*.

## Short-term memory

Multi-turn memory is provided by ovos-persona itself (the
`ovos-agents-short-term-memory-plugin`, enabled by default), **not** by this plugin. Each session's
conversation history is rebuilt and passed to `continue_chat` / `stream_sentences`, so the chat engine
remains stateless. To pick a different memory backend, set `"memory_module"` in the persona config (or
`null` to disable memory).

## Requirements

This plugin uses the OPM **agents framework**. Solver plugins are deprecated, so a persona names the
chat engine `ovos-chat-openai-plugin`. Install `ovos-persona` next to the plugin with
`pip install --pre ovos-persona`.

## End-to-end test

`test/end2end/test_e2e_persona_pipeline.py` exercises this integration with
[ovoscope](https://github.com/OpenVoiceOS/ovoscope): it boots a MiniCroft instance with the persona
pipeline, points the chat engine at a local OpenAI-compatible stub server (no network, no key), drives a
real `recognizer_loop:utterance` through the pipeline, and asserts a `speak` message is produced and the
user turn is recorded in per-session memory.

---
[← Available plugins](plugins.md) · [Home](../README.md)
