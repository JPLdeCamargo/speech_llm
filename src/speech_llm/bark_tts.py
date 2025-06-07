from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE

from .node import Node


class BarkTTS(Node):
    def __init__(self):
        self.speaker = "v2/pt_speaker_1"
        self.gen_temp = 0.6

    def load(self):
        preload_models()

    def execute(self, input):
        silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

        sentences = nltk.sent_tokenize(input, language="portuguese")
        pieces = []
        for sentence in sentences:
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=self.speaker,
                temp=self.gen_temp,
                min_eos_p=0.05,  # this controls how likely the generation is to end
            )

            audio_array = semantic_to_waveform(
                semantic_tokens,
                history_prompt=self.speaker,
            )
            pieces += [audio_array, silence.copy()]
        return Audio(np.concatenate(pieces), rate=SAMPLE_RATE)
