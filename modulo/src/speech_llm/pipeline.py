from .bark_tts import BarkTTS
from .llama import Llama
from .whisper import Whisper


class Pipeline:
    def __init__(self, hf_token: str):
        self.nodes = [Whisper(), Llama(hf_token), BarkTTS()]

    def load_all_nodes(self):
        for node in self.nodes:
            node.load()

    def execute_pipeline(self, input):
        for node in self.nodes:
            input = node.execute(input)
        return input
