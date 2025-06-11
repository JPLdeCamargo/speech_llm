import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import evaluate
import pandas as pd

import numpy.typing as npt
import numpy as np
from typing import Literal
from datasets import Dataset
import warnings

from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)

from peft import PeftModel, PeftConfig
from wer import WER


def whisper(paths:npt.NDArray[np.str_], reference:npt.NDArray[np.str_], language:Literal['en', 'pt'], batch_size:int):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    peft_model_id = "jp003/whisper-large-v3-lora-cv-pt"
    task = "transcribe"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, device_map="auto"
    )

    model = PeftModel.from_pretrained(model, peft_model_id)
    processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        return_timestamps=True,
    )

    dataset = Dataset.from_dict({"paths":paths, "reference":reference})
    def handle_batch(batch):
        result = pd.DataFrame(pipe(list(batch["paths"]), generate_kwargs={"language": language, "forced_decoder_ids": forced_decoder_ids}))
        return {"wer":[WER.get_wer(reference=list(batch["reference"]), transcription=list(result["text"]))]}

    wers = dataset.map(handle_batch, batched=True, batch_size=batch_size, remove_columns=["paths", "reference"])
    return wers.with_format("pandas")["wer"].mean()


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)  
    batch_size = int(input("batch_size: "))
    cap = int(input("Total tests: "))
    # if cap > 0:
    #     crt_i = cap
    # test_df = pd.read_csv('librispeech.csv')
    # test_df["sentence"] = test_df["sentence"].str.upper()
    # test_df = test_df.iloc[:crt_i]
    test_df = pd.read_csv('arquivos/cv-corpus-19.0-2024-09-13/pt/test.tsv', sep='\t')
    test_df["path"] = test_df["path"].apply(lambda x: f"arquivos/cv-corpus-19.0-2024-09-13/pt/clips/{x}")
    test_df["sentence"] = test_df["sentence"].str.upper()
    if cap > 0:
        test_df = test_df.iloc[:cap]

    print(whisper(test_df["path"].values, test_df["sentence"].values, "pt", batch_size=batch_size))