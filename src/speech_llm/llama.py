from .node import Node

from huggingface_hub import login
import torch
import transformers


class Llama(Node):
    def __init__(self, hf_token):
        login(token=hf_token)

        self.instruction_prompt = """
            Olá, você deve atuar como um ajudante de um estudante que está
            atualmente tentando completar seu TCC sobre interação por fala com
            LLMs. É de extrema importancia que você apenas forneça outputs como
            em uma conversa normal, uma vez que eles serão passados para um
            sintetizador de fala, então por favor não adicione formatações que
            fujam de conversar faladas.
        """

        self.conversation_history = [
            {"role": "system", "content": self.instruction_prompt}
        ]

    def load(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
        )

    def execute(self, input: str) -> str:
        self.conversation_history.append({"role": "user", "content": input})

        output = self.pipeline(self.conversation_history)
        response = output[0]["generated_text"][-1]["content"]

        self.conversation_history.append({"role": "assistant", "content": response})

        return response
