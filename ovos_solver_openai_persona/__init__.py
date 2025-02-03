from typing import Optional, Iterable, List, Dict

from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver


class OpenAIPersonaSolver(OpenAIChatCompletionsSolver):
    """default "Persona" engine"""

    def __init__(self, config=None):
        # defaults to gpt-3.5-turbo
        super().__init__(config=config)
        self.system_prompt = config.get("system_prompt") or "You are a helpful assistant. You are creative, clever, and very friendly."

    def get_chat_history(self, persona=None):
        return super().get_chat_history(self.system_prompt)

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
        answer = super().get_spoken_answer(query, lang, units)
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer

    def stream_chat_utterances(self, messages: List[Dict[str, str]],
                               lang: Optional[str] = None,
                               units: Optional[str] = None) -> Iterable[str]:
        messages = [{"role": "system", "content": self.system_prompt }] + messages
        answer = super().stream_chat_utterances(messages, lang, units)
        yield from answer

# for ovos-persona
LLAMA_DEMO = {
  "name": "Remote LLama",
  "solvers": [
    "ovos-solver-openai-plugin"
  ],
  "ovos-solver-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "sk-xxxx"
  }
}


if __name__ == "__main__":
    bot = OpenAIPersonaSolver(LLAMA_DEMO["ovos-solver-openai-plugin"])
    #for utt in bot.stream_utterances("describe quantum mechanics in simple terms"):
    #    print(utt)
        #  Quantum mechanics is a branch of physics that studies the behavior of atoms and particles at the smallest scales.
        #  It describes how these particles interact with each other, move, and change energy levels.
        #  Think of it like playing with toy building blocks that represent particles.
        #  Instead of rigid structures, these particles can be in different energy levels or "states." Quantum mechanics helps scientists understand and predict these states, making it crucial for many fields like chemistry, materials science, and engineering.

    # Quantum mechanics is a branch of physics that deals with the behavior of particles on a very small scale, such as atoms and subatomic particles. It explores the idea that particles can exist in multiple states at once and that their behavior is not predictable in the traditional sense.
    print(bot.spoken_answer("what is the definition of computer", lang="en-US"))
    # O português Pedro Álvares Cabral encontrou o caminho marítimo para o Brasil em 1500. Ele foi o responsável por descobrir o litoral brasileiro, embora Cristóvão Colombo tenha chegado à América do Sul em 1498, cinco anos antes. Cabral desembarcou na atual costa de Alagoas, no Nordeste do Brasil.
