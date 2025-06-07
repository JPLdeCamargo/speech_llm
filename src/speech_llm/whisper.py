from .node import Node

from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from peft import PeftModel, PeftConfig
import torch


class Whisper(Node):
    def load(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        peft_model_id = "jp003/whisper-large-v3-lora-cv-pt"
        task = "transcribe"
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
        )

        model = PeftModel.from_pretrained(model, peft_model_id)
        processor = WhisperProcessor.from_pretrained(
            peft_config.base_model_name_or_path, language="pt", task=task
        )
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="pt", task=task
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            return_timestamps=True,
        )

    def execute(self, input: str) -> str:
        return self.pipe(
            input,
            generate_kwargs={
                "language": "pt",
                "forced_decoder_ids": self.forced_decoder_ids,
            },
        )
