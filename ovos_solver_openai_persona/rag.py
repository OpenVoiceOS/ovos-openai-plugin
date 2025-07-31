import json
import requests
from ovos_plugin_manager.solvers import ChatMessageSolver
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_utils.log import LOG
from typing import Optional, List, Iterable, Dict, Any


class RequestException(Exception):
    """Custom exception for API request errors."""
    pass


class OpenAIRAGSolver(ChatMessageSolver):
    """
    An OVOS Solver plugin that implements Retrieval Augmented Generation (RAG)
    by interacting with OpenAI or Persona Server backend for vector store search
    and then directly calling the Persona Server's chat completions endpoint
    with the augmented context.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None):
        """
        Initializes the PersonaServerRAGSolver.

        Args:
            config (dict): Configuration dictionary for the solver. Expected keys:
                - "api_url" (str): Base URL of the ovos-persona-server (e.g., "http://localhost:8337/v1").
                - "vector_store_id" (str): The ID of the vector store to query for RAG.
                - "max_num_results" (int, optional): Max number of chunks to retrieve from search. Defaults to 5.
                - "max_context_tokens" (int, optional): Max tokens for retrieved context in the LLM prompt. Defaults to 2000.
                - "system_prompt_template" (str, optional): Template for the RAG system prompt.
                                                          Must contain "{context}" and "{question}" placeholders.
                - "llm_model" (str, optional): The model name to use for chat completions on the Persona Server.
                - "key" (str, optional): API key for the Persona Server's chat completions endpoint.
                - "llm_temperature" (float, optional): Sampling temperature for LLM. Defaults to 0.7.
                - "llm_top_p" (float, optional): Top-p sampling for LLM. Defaults to 1.0.
                - "llm_max_tokens" (int, optional): Max tokens for LLM generation. Defaults to 500.
            translator (LanguageTranslator, optional): Language translator instance.
            detector (LanguageDetector, optional): Language detector instance.
            priority (int): Solver priority.
            enable_tx (bool): Enable translation.
            enable_cache (bool): Enable caching.
            internal_lang (str, optional): Internal language code.

        Raises:
            ValueError: If required configuration parameters are missing or invalid.
        """
        super().__init__(config=config, translator=translator,
                         detector=detector, priority=priority,
                         enable_tx=enable_tx, enable_cache=enable_cache,
                         internal_lang=internal_lang)

        # Persona Server RAG Configuration
        self.api_url = self.config.get("api_url")
        self.vector_store_id = self.config.get("vector_store_id")
        self.max_num_results = self.config.get("max_num_results", 5)
        self.max_context_tokens = self.config.get("max_context_tokens", 2000)

        if not self.api_url:
            raise ValueError("api_url must be set in config for PersonaServerRAGSolver")
        if not self.vector_store_id:
            raise ValueError("vector_store_id must be set in config for PersonaServerRAGSolver")

        # RAG System Prompt Template
        self.system_prompt_template = self.config.get("system_prompt_template")
        if not self.system_prompt_template:
            self.system_prompt_template = (
                "You are a helpful assistant. Use the following context to answer the user's question. "
                "If the answer is not in the context, state that you don't know.\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}"
            )
            LOG.debug(f"system_prompt_template not set, defaulting to: '{self.system_prompt_template}'")
        elif "{context}" not in self.system_prompt_template or "{question}" not in self.system_prompt_template:
            raise ValueError("system_prompt_template must contain '{context}' and '{question}' placeholders.")

        # LLM Parameters for direct call to Persona Server's chat completions
        self.llm_model = self.config.get("llm_model")
        self.key = self.config.get("key") # This is the key for the Persona Server's chat endpoint
        self.llm_temperature = self.config.get("llm_temperature", 0.7)
        self.llm_top_p = self.config.get("llm_top_p", 1.0)
        self.llm_max_tokens = self.config.get("llm_max_tokens", 500)

        if not self.llm_model:
             LOG.warning("llm_model not set. This is fine for Persona Server, but ensure your LLM provider allows it")
        # key can be optional for local Ollama setups, but good practice to include check
        if not self.key:
             LOG.warning("key not set. This might be fine for local Ollama, but ensure your Persona Server allows unauthenticated access or provide a key.")

        # Memory for this RAG solver
        self.memory = config.get("enable_memory", True)
        self.max_utts = config.get("memory_size", 3)
        self.qa_pairs = []  # Stores (user_query, final_rag_answer) for history

    def _search_vector_store(self, query: str) -> List[str]:
        """
        Searches the configured vector store for relevant text chunks matching the user query.

        Parameters:
            query (str): The user's query string to search for relevant context.

        Returns:
            List[str]: A list of text chunks retrieved from the vector store that are relevant to the query.

        Raises:
            RequestException: If the search request fails or the response format is invalid.
        """
        search_url = f"{self.api_url}/vector_stores/{self.vector_store_id}/search"
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "max_num_results": self.max_num_results
        }
        LOG.debug(f"Sending RAG search request to {search_url} with query: {query}")

        try:
            response = requests.post(search_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if "data" not in data or not isinstance(data["data"], list):
                raise RequestException(f"Unexpected response format from RAG search: {data}")

            # Extract content from search results
            retrieved_chunks = [item["content"] for item in data["data"] if "content" in item]
            LOG.debug(f"Retrieved {len(retrieved_chunks)} chunks from vector store.")
            return retrieved_chunks
        except requests.exceptions.RequestException as e:
            LOG.error(f"Error during RAG search request to {search_url}: {e}")
            raise RequestException(f"Failed to retrieve context from vector store: {e}")
        except json.JSONDecodeError as e:
            LOG.error(f"Failed to parse JSON response from RAG search: {e}")
            raise RequestException(f"Invalid JSON response from vector store: {e}")

    def _build_llm_messages(self, user_query: str, retrieved_context_chunks: List[str], chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Constructs the message list for the LLM by combining retrieved context, recent chat history, and the current user query.

        The method concatenates relevant context chunks (up to a token limit), formats the system prompt with this context and the user's question, appends recent Q&A pairs from memory, and adds the current user query as the final message.

        Parameters:
            user_query (str): The user's current question or utterance.
            retrieved_context_chunks (List[str]): Relevant text segments retrieved from the vector store.
            chat_history (List[Dict[str, str]]): Previous conversation history.

        Returns:
            List[Dict[str, str]]: The complete list of messages to send to the LLM, including system prompt, chat history, and user query.
        """
        context_str = ""
        current_context_tokens = 0

        # Build context string, respecting max_context_tokens
        for chunk in retrieved_context_chunks:
            # Estimate tokens for the chunk if added to a system prompt
            chunk_tokens = len(chunk.split()) # Simple word count as token estimate
            if current_context_tokens + chunk_tokens <= self.max_context_tokens:
                context_str += chunk + "\n\n"
                current_context_tokens += chunk_tokens
            else:
                LOG.debug(f"Truncating RAG context due to max_context_tokens limit. Added {current_context_tokens} tokens.")
                break

        # Construct the RAG-augmented system prompt
        rag_system_prompt = self.system_prompt_template.format(
            context=context_str.strip(),
            question=user_query
        )

        # Start with the RAG-augmented system prompt
        messages: List[Dict[str, str]] = [{"role": "system", "content": rag_system_prompt}]

        # Append prior conversation history
        for q, a in self.qa_pairs[-1 * self.max_utts:]: # Use RAG solver's memory
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        # Append the current user query
        messages.append({"role": "user", "content": user_query})

        LOG.debug(f"Constructed LLM prompt messages: {messages}")
        return messages

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Return the recent chat history as a list of user and assistant messages.

        Returns:
            List of message dictionaries representing the most recent question-answer pairs, formatted with roles 'user' and 'assistant'.
        """
        # The base class expects a list of messages (role, content).
        # We store (query, answer) tuples.
        history_messages = []
        for q, a in self.qa_pairs[-1 * self.max_utts:]:
            history_messages.append({"role": "user", "content": q})
            history_messages.append({"role": "assistant", "content": a})
        return history_messages

    ## chat completions api - message list as input
    def continue_chat(self, messages: List[Dict[str, str]],
                      lang: Optional[str],
                      units: Optional[str] = None) -> Optional[str]:
        """
        Generate a chat response by augmenting the user query with retrieved context from a vector store and sending the constructed prompt to the Persona Server's chat completions endpoint.

        Parameters:
            messages (List[Dict[str, str]]): List of chat messages, where the last message is treated as the current user query.
            lang (Optional[str]): Optional language code for the response.
            units (Optional[str]): Optional unit system for numerical values.

        Returns:
            Optional[str]: The generated response as a string, or None if no valid response is produced.

        Raises:
            RequestException: If the Persona Server's chat completions endpoint returns an error or an invalid response.
        """
        user_query = messages[-1]["content"] # Get the current user query

        # 1. Search vector store for context
        try:
            retrieved_chunks = self._search_vector_store(user_query)
        except RequestException:
            LOG.warning("RAG search failed, proceeding with LLM without augmented context.")
            retrieved_chunks = []

        # 2. Build augmented messages for the LLM, including RAG solver's history
        augmented_messages = self._build_llm_messages(user_query, retrieved_chunks, messages[:-1]) # Pass existing history

        # 3. Call Persona Server's chat completions endpoint
        chat_completions_url = f"{self.api_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"

        payload = {
            "model": self.llm_model,
            "messages": augmented_messages,
            "max_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature,
            "top_p": self.llm_top_p,
            "stream": False # Non-streaming call
        }
        LOG.debug(f"Sending LLM request to {chat_completions_url} with payload: {payload}")

        try:
            response = requests.post(chat_completions_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0 and "message" in data["choices"][0]:
                answer = data["choices"][0]["message"]["content"]
                if self.memory and answer:
                    self.qa_pairs.append((user_query, answer)) # Store for future turns
                return answer
            else:
                raise RequestException(f"Unexpected response format from LLM: {data}")
        except requests.exceptions.RequestException as e:
            LOG.error(f"Error during LLM chat completions request to {chat_completions_url}: {e}")
            raise RequestException(f"Failed to get LLM response: {e}")
        except json.JSONDecodeError as e:
            LOG.error(f"Failed to parse JSON response from LLM: {e}")
            raise RequestException(f"Invalid JSON response from LLM: {e}")

    def stream_chat_utterances(self, messages: List[Dict[str, str]],
                               lang: Optional[str] = None,
                               units: Optional[str] = None) -> Iterable[str]: # Yields raw data: lines
        """
        Streams chat completion responses from the Persona Server using Retrieval Augmented Generation (RAG), yielding each line of streamed data as it arrives.

        The method retrieves relevant context from the vector store based on the latest user query, augments the chat history, and streams the LLM's response line by line. If enabled, it stores the full answer in memory for multi-turn conversations.

        Parameters:
            messages (List[Dict[str, str]]): The chat history, with the last message as the current user query.
            lang (Optional[str]): Optional language code for the query.
            units (Optional[str]): Optional units for the query.

        Returns:
            Iterable[str]: Yields each raw data line (as a string) from the streaming API response.
        """
        user_query = messages[-1]["content"] # Get the current user query

        # 1. Search vector store for context
        try:
            retrieved_chunks = self._search_vector_store(user_query)
        except RequestException:
            LOG.warning("RAG search failed, proceeding with LLM without augmented context.")
            retrieved_chunks = []

        # 2. Build augmented messages for the LLM, including RAG solver's history
        augmented_messages = self._build_llm_messages(user_query, retrieved_chunks, messages[:-1]) # Pass existing history

        # 3. Call Persona Server's chat completions endpoint in streaming mode
        chat_completions_url = f"{self.api_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"

        payload = {
            "model": self.llm_model,
            "messages": augmented_messages,
            "max_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature,
            "top_p": self.llm_top_p,
            "stream": True # Streaming call
        }
        LOG.debug(f"Sending streaming LLM request to {chat_completions_url} with payload: {payload}")

        full_answer = "" # To reconstruct the full answer for memory
        try:
            with requests.post(chat_completions_url, headers=headers, data=json.dumps(payload), stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        # The Persona Server already sends "data: " prefix and "[DONE]"
                        # So we can yield the line directly
                        if decoded_line.startswith("data: "):
                            json_part = decoded_line[len("data: "):].strip()
                            if json_part == "[DONE]":
                                break
                            try:
                                chunk_dict = json.loads(json_part)
                                content = chunk_dict.get("choices", [{}])[0].get("delta", {}).get("content")
                                if content:
                                    full_answer += content
                            except json.JSONDecodeError:
                                pass # Ignore non-JSON lines
                        yield decoded_line # Yield the raw data: line
        except requests.exceptions.RequestException as e:
            LOG.error(f"Error during streaming LLM chat completions request to {chat_completions_url}: {e}")
            # Yield an error chunk in the stream
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        if self.memory and full_answer:
            self.qa_pairs.append((user_query, full_answer)) # Store for future turns


    ## completions api - single text as input (delegates to chat)
    def stream_utterances(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> Iterable[str]:
        """
        Streams the assistant's response for a given user query, incorporating current chat history and Retrieval Augmented Generation context.

        Parameters:
            query (str): The user's input query.
            lang (Optional[str]): Language code for the response, if applicable.
            units (Optional[str]): Units relevant to the query, if applicable.

        Returns:
            Iterable[str]: Yields raw data chunks from the streaming chat completions API.
        """
        # For stream_utterances, we directly build a single-turn message list
        # We need to include existing chat history here as well for proper context
        messages: List[Dict[str, str]] = self.get_chat_history()
        messages.append({"role": "user", "content": query})
        yield from self.stream_chat_utterances(messages, lang, units)

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> Optional[str]:
        """
        Return the assistant's spoken answer to a user query, incorporating recent chat history for context.

        Parameters:
            query (str): The user's input question.
            lang (Optional[str]): Language code for the response, if specified.
            units (Optional[str]): Units relevant to the query, if specified.

        Returns:
            Optional[str]: The assistant's text response, or None if no answer is generated.
        """
        # For get_spoken_answer, we need to include existing chat history
        messages: List[Dict[str, str]] = self.get_chat_history()
        messages.append({"role": "user", "content": query})
        return self.continue_chat(messages=messages, lang=lang, units=units)

    def get_messages(self, utt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Return the current chat history messages with the latest user utterance appended.

        Parameters:
        	utt (str): The current user utterance to add to the message list.

        Returns:
        	List of message dictionaries representing the conversation history plus the new user message.
        """
        messages = self.get_chat_history()
        messages.append({"role": "user", "content": utt})
        return messages


if __name__ == "__main__":
    # --- Live Test Example for PersonaServerRAGSolver ---
    # This example assumes:
    # 1. ovos-persona-server is running at http://localhost:8337
    # 2. You have a vector store created and populated with documents
    #    (e.g., using the curl examples from the README)

    # --- IMPORTANT: Replace these with your actual values ---
    PERSONA_SERVER_URL = "http://0.0.0.0:8337/v1"
    VECTOR_STORE_ID = "vs_YgqHwhmyJ48kkI6jU6G3ApSq"  # <<< REPLACE THIS with your vector store ID
    LLM_API_KEY = "sk-xxxx"  # Can be any non-empty string for local setups like Ollama
    LLM_MODEL = "qwen2.5:7b" # Or "gpt-4o-mini", etc.
    # ----------------------------------------------------

    rag_solver_config = {
        "api_url": PERSONA_SERVER_URL,
        "key": LLM_API_KEY,
        "llm_model": LLM_MODEL,
        "vector_store_id": VECTOR_STORE_ID,
        "max_num_results": 3,  # Max chunks to retrieve
        "max_context_tokens": 1000,  # Max tokens for the context in the LLM prompt
        "system_prompt_template": (
            "You are a helpful assistant. Use the following context to answer the user's question. "
            "You can give personal opinions and deviate from the goal, but keep things factual."
            "If the answer is not in the context and can not be inferred from the conversation history, state that you don't know.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
        "llm_temperature": 0.7,
        "llm_top_p": 1.0,
        "llm_max_tokens": 500,
        "enable_memory": True, # Enable memory for this solver
        "memory_size": 3 # Store 3 Q&A pairs
    }

    print("--- Initializing PersonaServerRAGSolver ---")
    try:
        rag_solver = OpenAIRAGSolver(config=rag_solver_config)
        print("Solver initialized successfully.")
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"Error initializing RAG solver: {e}")
        print("Please ensure your configuration (api_url, vector_store_id, llm_model, key) is correct.")
        exit(1)

    print("\n--- Testing get_spoken_answer (non-streaming) ---")
    test_query_non_streaming = "What is Nedzo"
    print(f"Query: {test_query_non_streaming}")
    try:
        answer = rag_solver.get_spoken_answer(query=test_query_non_streaming, lang="en")
        print(f"Answer: {answer}")
    except RequestException as e:
        print(f"Error during non-streaming RAG query: {e}")
        print("Please ensure ovos-persona-server is running and your vector_store_id is valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- Testing stream_utterances (streaming) ---")
    test_query_streaming = "what is the purpose and goal"
    print(f"Query: {test_query_streaming}")
    print("Streaming Answer: ", end="")
    try:
        for chunk_line in rag_solver.stream_utterances(query=test_query_streaming, lang="en"):
            # The Persona Server sends "data: {json_chunk}\n\n" or "data: [DONE]\n\n"
            # We need to parse the JSON and extract content
            if chunk_line.startswith("data: "):
                json_part = chunk_line[len("data: "):].strip()
                if json_part == "[DONE]":
                    break
                try:
                    chunk_dict = json.loads(json_part)
                    content = chunk_dict.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content:
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    # Handle cases where the line is not valid JSON (e.g., just "data: ")
                    pass
        print()  # Newline after streaming finishes
    except RequestException as e:
        print(f"\nError during streaming RAG query: {e}")
        print("Please ensure ovos-persona-server is running and your vector_store_id is valid.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("\n--- Testing with a query that might not find context ---")
    test_query_no_context = "What color is the sky on Mars?"
    print(f"Query: {test_query_no_context}")
    try:
        answer_no_context = rag_solver.get_spoken_answer(query=test_query_no_context, lang="en")
        print(f"Answer: {answer_no_context}")
    except RequestException as e:
        print(f"Error during RAG query (no context): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- Testing multi-turn conversation with memory ---")
    print("Query 1:What is the role of agents?")
    try:
        answer1 = rag_solver.get_spoken_answer(query="What is the role of agents", lang="en")
        print(f"Answer 1: {answer1}")
    except Exception as e:
        print(f"Error in multi-turn query 1: {e}")

    print("\nQuery 2: how many of them must be supported at once?")
    try:
        answer2 = rag_solver.get_spoken_answer(query="how many of them must be supported at once?", lang="en")
        print(f"Answer 2: {answer2}")
    except Exception as e:
        print(f"Error in multi-turn query 2: {e}")

    print("\nQuery 3: summarize our conversation so far")
    try:
        answer3 = rag_solver.get_spoken_answer(query="summarize our conversation so far", lang="en")
        print(f"Answer 3: {answer3}")
    except Exception as e:
        print(f"Error in multi-turn query 3: {e}")
