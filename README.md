# <img src='https://raw.githack.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> OVOS OpenAI Plugin

This plugin is designed to leverage the **OpenAI API** for various functionalities within the OpenVoiceOS ecosystem. It provides a set of OVOS plugins that interact with OpenAI's services. Crucially, it is also compatible with **self-hosted OpenAI-compatible alternatives**, such as the [OVOS Persona Server](https://github.com/OpenVoiceOS/ovos-persona-server), or any other project that implements the full suite of OpenAI API endpoints (Chat Completions, Embeddings, Files, and Vector Stores). This flexibility allows you to choose between cloud-based OpenAI services or a local, private setup.

Specifically, this plugin provides:

  - `ovos-solver-openai-plugin` for general chat completions, primarily for usage with [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) (and in older ovos releases with [ovos-skill-fallback-chatgpt](https://www.google.com/search?q=))
  - `ovos-solver-openai-rag-plugin` for Retrieval Augmented Generation using a compatible backend (like `ovos-persona-server`) as a knowledge source.
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
    "system_prompt": "You are helping assistant who gives very short and factual answers in maximum twenty words and you don't use emojis",
    "model": "llama3.1:8b"
  }
}
```

Then say "Chat with {name_from_json}" to enable it, more details can be found in [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) README

This plugins also provides a default "Remote LLama" demo persona, it points to a public server hosted by @goldyfruit.

-----

## RAG Solver Usage

The `ovos-solver-openai-rag-plugin` enables **Retrieval Augmented Generation (RAG)**. This means your OVOS assistant can answer questions by first searching for relevant information in a configured knowledge base (a "vector store" hosted by a compatible backend like `ovos-persona-server`), and then using an LLM to generate a coherent answer based on that retrieved context.

This is particularly useful for:

  * Answering questions about specific documentation, personal notes, or proprietary data.
  * Reducing LLM hallucinations by grounding responses in factual, provided information.

### How it Works

1.  **Search**: When a user asks a question, the RAG solver first sends the query to the configured backend's vector store search endpoint.
2.  **Retrieve**: The backend returns relevant text chunks (documents or passages) from your indexed data.
3.  **Augment**: These retrieved chunks are then injected into the LLM's prompt, along with the user's original query and conversation history.
4.  **Generate**: The LLM processes this augmented prompt and generates an answer, prioritizing the provided context.

### Configuration

To use the RAG solver, you need to configure it in your `~/.config/ovos_persona/llm.json` file. You will need:

1.  A **compatible OpenAI API backend running** (e.g., [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server)) with a populated vector store.
2.  The `vector_store_id` of your created vector store on that backend.
3.  The `llm_model` and `llm_api_key` for the LLM that your chosen backend will use for chat completions.

Here's an example `llm.json` configuration for a RAG persona:

```json
{
  "name": "My RAG Assistant",
  "solvers": [
    "ovos-solver-openai-rag-plugin"
  ],
  "ovos-solver-openai-rag-plugin": {
    "persona_server_url": "http://localhost:8337/v1",  // URL of your OpenAI-compatible backend
    "vector_store_id": "vs_your_vector_store_id_here", // <<< REPLACE THIS!
    "max_num_results": 5,                             // Max text chunks to retrieve
    "max_context_tokens": 2000,                       // Max tokens from retrieved context for LLM
    "system_prompt_template": "You are a helpful assistant. Use the following context to answer the user's question. If the answer is not in the context, state that you don't know.\n\nContext:\n{context}\n\nQuestion:\n{question}",
    "llm_model": "llama3.1:8b",                       // The LLM model name used by the backend
    "llm_api_key": "sk-xxxx",                         // API key for the LLM on the backend (can be dummy for local setups)
    "llm_temperature": 0.7,
    "llm_top_p": 1.0,
    "llm_max_tokens": 500,
    "enable_memory": true,                            // Enable conversation history for RAG
    "memory_size": 3                                  // Number of Q&A pairs to remember
  }
}
```

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
