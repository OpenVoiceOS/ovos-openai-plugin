import json
from typing import Optional, Iterable, List, Dict

import requests
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_plugin_manager.templates.solvers import ChatMessageSolver
from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_utils.log import LOG
from requests import RequestException

MessageList = List[Dict[str, str]]  # for typing


class OpenAICompletionsSolver(QuestionSolver):
    def __init__(self, config=None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None):
        """
        Initializes the OpenAICompletionsSolver with API configuration and credentials.
         
        Raises:
            ValueError: If the API key is not provided in the configuration.
        """
        super().__init__(config=config, translator=translator,
                 detector=detector, priority=priority,
                 enable_tx=enable_tx, enable_cache=enable_cache,
                 internal_lang=internal_lang)
        self.api_url = f"{self.config.get('api_url', 'https://api.openai.com/v1')}/completions"
        self.engine = self.config.get("model", "gpt-4o-mini")
        self.key = self.config.get("key")
        if not self.key:
            LOG.error("key not set in config")
            raise ValueError("key must be set")

    # OpenAI API integration
    def _do_api_request(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key
        }

        # https://platform.openai.com/docs/api-reference/completions/create
        payload = {
            "model": self.engine,
            "prompt": prompt,
            "max_tokens": self.config.get("max_tokens", 100),
            "temperature": self.config.get("temperature", 0.5),
            # between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
            "top_p": self.config.get("top_p", 0.2),
            # nucleus sampling alternative to temperature, the model considers the results of the tokens with top_p probability mass. 0.1 means only tokens comprising top 10% probability mass are considered.
            "n": 1,  # How many completions to generate for each prompt.
            "frequency_penalty": self.config.get("frequency_penalty", 0),
            # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            "presence_penalty": self.config.get("presence_penalty", 0),
            # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
            "stop": self.config.get("stop_token")
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload)).json()
        if "error" in response:
            raise RequestException(response["error"])
        return response["choices"][0]["text"]

    # officially exported Solver methods
    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> Optional[str]:
        """
        Obtain the spoken answer for a given query.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The spoken answer as a text response.
        """
        response = self._do_api_request(query)
        answer = response.strip()
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer


def post_process_sentence(text: str) -> str:
    text = text.replace("*", "")  # TTS often literally reads "asterisk"
    # TODO - option to drop emojis etc.
    return text.strip()


class OpenAIChatCompletionsSolver(ChatMessageSolver):
    def __init__(self, config=None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 25,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None):
        """
        Initializes the OpenAIChatCompletionsSolver with API configuration, memory settings, and system prompt.
         
        Raises:
            ValueError: If the API key is not provided in the configuration.
        """
        super().__init__(config=config, translator=translator,
                 detector=detector, priority=priority,
                 enable_tx=enable_tx, enable_cache=enable_cache,
                 internal_lang=internal_lang)
        self.api_url = f"{self.config.get('api_url', 'https://api.openai.com/v1')}/chat/completions"
        self.engine = self.config.get("model", "gpt-4o-mini")
        self.key = self.config.get("key")
        if not self.key:
            LOG.error("key not set in config")
            raise ValueError("key must be set")
        self.memory = config.get("enable_memory", True)
        self.max_utts = config.get("memory_size", 3)
        self.qa_pairs = []  # tuple of q+a
        if "persona" in config:
            LOG.warning("'persona' config option is deprecated, use 'system_prompt' instead")
        if "initial_prompt" in config:
            LOG.warning("'initial_prompt' config option is deprecated, use 'system_prompt' instead")
        self.system_prompt = config.get("system_prompt") or config.get("initial_prompt")
        if not self.system_prompt:
            self.system_prompt =  "You are a helpful assistant."
            LOG.error(f"system prompt not set in config! defaulting to '{self.system_prompt}'")

    # OpenAI API integration
    def _do_api_request(self, messages):
        """
        Sends a chat completion request to the OpenAI API and returns the assistant's reply.
        
        Args:
            messages: A list of message dictionaries representing the conversation history.
        
        Returns:
            The content of the assistant's reply as a string.
        
        Raises:
            RequestException: If the OpenAI API returns an error in the response.
        """
        s = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key
        }

        # params docs
        # https://platform.openai.com/docs/api-reference/completions/create
        payload = {
            "model": self.engine,
            "messages": messages,
            "max_tokens": self.config.get("max_tokens", 100),
            "temperature": self.config.get("temperature", 0.5),
            # between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
            "top_p": self.config.get("top_p", 0.2),
            # nucleus sampling alternative to temperature, the model considers the results of the tokens with top_p probability mass. 0.1 means only tokens comprising top 10% probability mass are considered.
            "n": 1,  # How many completions to generate for each prompt.
            "frequency_penalty": self.config.get("frequency_penalty", 0),
            # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            "presence_penalty": self.config.get("presence_penalty", 0),
            # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
            "stop": self.config.get("stop_token")
        }
        response = s.post(self.api_url, headers=headers, data=json.dumps(payload)).json()
        if "error" in response:
            raise RequestException(response["error"])
        return response["choices"][0]["message"]["content"]

    def _do_streaming_api_request(self, messages):

        """
        Streams response content from the OpenAI chat completions API.
        
        Sends a POST request with the provided chat messages and yields content chunks as they are received from the streaming API. Stops iteration if an error is encountered or the response is finished.
        
        Args:
            messages: A list of chat message dictionaries to send as context.
        
        Yields:
            str: Segments of the assistant's reply as they arrive from the API.
        """
        s = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key
        }

        # params docs
        # https://platform.openai.com/docs/api-reference/completions/create
        payload = {
            "model": self.engine,
            "messages": messages,
            "max_tokens": self.config.get("max_tokens", 100),
            "temperature": self.config.get("temperature", 0.5),
            # between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
            "top_p": self.config.get("top_p", 0.2),
            # nucleus sampling alternative to temperature, the model considers the results of the tokens with top_p probability mass. 0.1 means only tokens comprising top 10% probability mass are considered.
            "n": 1,  # How many completions to generate for each prompt.
            "frequency_penalty": self.config.get("frequency_penalty", 0),
            # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            "presence_penalty": self.config.get("presence_penalty", 0),
            # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
            "stop": self.config.get("stop_token"),
            "stream": True
        }
        for chunk in s.post(self.api_url, headers=headers,
                            stream=True, data=json.dumps(payload)).iter_lines():
            if chunk:
                chunk = chunk.decode("utf-8")
                chunk = json.loads(chunk.split("data: ", 1)[-1])
                if "error" in chunk and "message" in chunk["error"]:
                    LOG.error("API returned an error: " + chunk["error"]["message"])
                    break
                if chunk["choices"][0].get("finish_reason"):
                    break
                if "content" not in chunk["choices"][0]["delta"]:
                    continue
                yield chunk["choices"][0]["delta"]["content"]

    def get_chat_history(self, system_prompt=None):
        """
        Builds the chat history as a list of messages, starting with a system prompt.
        
        Args:
            system_prompt: Optional override for the system prompt message.
        
        Returns:
            A list of message dictionaries representing the system prompt and the most recent user-assistant exchanges.
        """
        qa = self.qa_pairs[-1 * self.max_utts:]
        system_prompt = system_prompt or self.system_prompt or "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for q, a in qa:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        return messages

    def get_messages(self, utt, system_prompt=None) -> MessageList:
        """
        Builds a list of chat messages including the system prompt, recent conversation history, and the current user utterance.
        
        Args:
        	utt: The current user input to be appended as the latest message.
        	system_prompt: Optional system prompt to use as the initial message.
        
        Returns:
        	A list of message dictionaries representing the chat context for the API.
        """
        messages = self.get_chat_history(system_prompt)
        messages.append({"role": "user", "content": utt})
        return messages

    # abstract Solver methods
    def continue_chat(self, messages: MessageList,
                      lang: Optional[str],
                      units: Optional[str] = None) -> Optional[str]:
        """
        Generates a chat response using the provided message history and updates memory if enabled.

        If the first message is not a system prompt, prepends the system prompt. Processes the API response and returns a cleaned answer, or None if the answer is empty or only punctuation/underscores. Updates internal memory with the latest question and answer if memory is enabled.

        Args:
            messages: List of chat messages with 'role' and 'content' keys.
            lang: Optional language code for the response.
            units: Optional unit system for numerical values.

        Returns:
            The generated response as a string, or None if no valid response is produced.
        """
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": self.system_prompt }] + messages
        response = self._do_api_request(messages)
        answer = post_process_sentence(response)
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        if self.memory:
            query = messages[-1]["content"]
            self.qa_pairs.append((query, answer))
        return answer

    def stream_chat_utterances(self, messages: MessageList,
                               lang: Optional[str] = None,
                               units: Optional[str] = None) -> Iterable[str]:
        """
        Stream utterances for the given chat history as they become available.

        Args:
            messages: The chat messages.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            Iterable[str]: An iterable of utterances.
        """
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": self.system_prompt }] + messages
        answer = ""
        query = messages[-1]["content"]
        if self.memory:
            self.qa_pairs.append((query, answer))

        for chunk in self._do_streaming_api_request(messages):
            answer += chunk
            if any(chunk.endswith(p) for p in [".", "!", "?", "\n", ":"]):
                if len(chunk) >= 2 and chunk[-2].isdigit() and chunk[-1] == ".":
                    continue  # dont split numbers
                if answer.strip():
                    if self.memory:
                        full_ans = f"{self.qa_pairs[-1][-1]}\n{answer}".strip()
                        self.qa_pairs[-1] = (query, full_ans)
                    yield post_process_sentence(answer)
                answer = ""

    def stream_utterances(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> Iterable[str]:
        """
        Stream utterances for the given query as they become available.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            Iterable[str]: An iterable of utterances.
        """
        messages = self.get_messages(query)
        yield from self.stream_chat_utterances(messages, lang, units)

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> Optional[str]:
        """
        Obtain the spoken answer for a given query.

        Args:
            query (str): The query text.
            lang (Optional[str]): Optional language code. Defaults to None.
            units (Optional[str]): Optional units for the query. Defaults to None.

        Returns:
            str: The spoken answer as a text response.
        """
        messages = self.get_messages(query)
        # just for api compat since it's a subclass, shouldn't be directly used
        return self.continue_chat(messages=messages, lang=lang, units=units)
