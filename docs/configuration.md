# Configuration reference

All plugins in this package share the same OpenAI client configuration. Each plugin reads the keys it
needs from its own config dictionary.

## Common keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_url` | str | `https://api.openai.com/v1` | Base URL of the OpenAI-compatible server (the `/v1` root). `/chat/completions` is appended automatically. |
| `key` | str | `""` | API key sent as `Authorization: Bearer <key>`. Optional for local servers that do not require authentication. |
| `model` | str | `gpt-4o-mini` | Model identifier passed to the server. |
| `system_prompt` | str | *(plugin specific)* | System instruction prepended to the conversation. |

## Generation parameters

These are forwarded to the Chat Completions request body:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_tokens` | int | `300` | Maximum number of tokens to generate. |
| `temperature` | float | `0.5` | Sampling temperature (0–2). |
| `top_p` | float | `0.2` | Nucleus sampling probability mass. |
| `frequency_penalty` | float | `0` | Penalize repeated tokens (−2 to 2). |
| `presence_penalty` | float | `0` | Encourage new topics (−2 to 2). |
| `stop_token` | str / list | `None` | Stop sequence(s). |

## Chat engine specific keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `allow_system_prompts` | bool | `false` | If `false`, system messages supplied by the caller are stripped and replaced by `system_prompt`. If `true`, a caller-supplied system message is merged with `system_prompt`. |

## Summarizer specific keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `prompt_template` | str | built-in | Template containing `{content}`, filled with the document to summarize. |

## Example

```json
{
  "api_url": "https://api.openai.com/v1",
  "key": "sk-xxxx",
  "model": "gpt-4o-mini",
  "system_prompt": "You are a concise assistant.",
  "max_tokens": 300,
  "temperature": 0.5
}
```
