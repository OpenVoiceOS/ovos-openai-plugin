import warnings
from ovos_solver_openai_persona.engines import OpenAIChatCompletionsSolver

class OpenAIPersonaSolver(OpenAIChatCompletionsSolver):
    def __init__(self, *args, **kwargs):
           """
           Initializes the solver and issues a deprecation warning.
           
           A DeprecationWarning is raised advising to use OpenAIChatCompletionsSolver instead.
           """
           warnings.warn(
              "use OpenAIChatCompletionsSolver instead",
              DeprecationWarning,
              stacklevel=2,
           )
           super().__init__(*args, **kwargs)

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
    bot = OpenAIChatCompletionsSolver(LLAMA_DEMO["ovos-solver-openai-plugin"])
    #for utt in bot.stream_utterances("describe quantum mechanics in simple terms"):
    #    print(utt)
        #  Quantum mechanics is a branch of physics that studies the behavior of atoms and particles at the smallest scales.
        #  It describes how these particles interact with each other, move, and change energy levels.
        #  Think of it like playing with toy building blocks that represent particles.
        #  Instead of rigid structures, these particles can be in different energy levels or "states." Quantum mechanics helps scientists understand and predict these states, making it crucial for many fields like chemistry, materials science, and engineering.

    # Quantum mechanics is a branch of physics that deals with the behavior of particles on a very small scale, such as atoms and subatomic particles. It explores the idea that particles can exist in multiple states at once and that their behavior is not predictable in the traditional sense.
    print(bot.spoken_answer("what is the definition of computer", lang="en-US"))
    # O português Pedro Álvares Cabral encontrou o caminho marítimo para o Brasil em 1500. Ele foi o responsável por descobrir o litoral brasileiro, embora Cristóvão Colombo tenha chegado à América do Sul em 1498, cinco anos antes. Cabral desembarcou na atual costa de Alagoas, no Nordeste do Brasil.
