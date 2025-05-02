# <img src='https://raw.githack.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> OVOS OpenAI Plugin

Leverages [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create) to provide the following ovos plugins: 
- `ovos-solver-openai-plugin` for usage with [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) (and in older ovos releases with [ovos-skill-fallback-chatgpt]())
- `ovos-dialog-transformer-openai-plugin` to rewrite OVOS dialogs just before TTS executes in [ovos-audio](https://github.com/OpenVoiceOS/ovos-audio)
- `ovos-summarizer-openai-plugin` to summarize text, not used directly but provided for consumption by other plugins/skills

## Install

`pip install ovos-openai-plugin`

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

## Dialog Transformer

you can rewrite text dynamically based on specific personas, such as simplifying explanations or mimicking a specific tone.  

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

## Remote Persona / Proxies

You can run any persona behind a OpenAI compatible server via [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server). 

This allows you to offload the workload to a standalone server, either for performance reasons or to keep api keys in a single safe place.

Then just configure this plugin to point to your persona server like it was OpenAI

