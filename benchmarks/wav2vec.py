from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset
import pandas as pd
import torch
import soundfile as sf
import numpy.typing as npt
import numpy as np
from typing import Literal
from wer import WER
from audio_loader import AudioLoader


def wav2vec(paths:npt.NDArray[np.str_], reference:npt.NDArray[np.str_], language:Literal['en', 'pt'], batch_size:int):
    # load model and tokenizer
    model_id = None
    if language == 'en':
        model_id = "facebook/wav2vec2-large-960h-lv60-self"
    elif language == 'pt':
        model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model = model.to(device)
    print("model loaded")
        
    dataset = Dataset.from_dict({"paths":paths, "reference":reference})

    def handle_batch(batch):
        audio_array = AudioLoader.load(batch["paths"], 16000, as_list=True)
        audio_array = [np.squeeze(audio) for audio in audio_array]
        inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt", batch_size=batch_size, padding=True).to(device).input_values

        with torch.no_grad():
            outputs = model(inputs).logits

        ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(ids)
        return {"wer":[WER.get_wer(reference=list(batch["reference"]), transcription=list(transcription))]}


    wers = dataset.map(handle_batch, batched=True, batch_size=batch_size, remove_columns=["paths", "reference"])
    return wers.with_format("pandas")["wer"].mean()


if __name__ == "__main__":
    batch_size = int(input("batch_size: "))
    cap = int(input("Total tests: "))
    if cap > 0:
        crt_i = cap
    # test_df = pd.read_csv('librispeech.csv')
    test_df = pd.read_csv('data/cv-corpus-19.0-2024-09-13/pt/test.tsv', sep='\t')
    test_df["path"] = test_df["path"].apply(lambda x: f"data/cv-corpus-19.0-2024-09-13/pt/clips/{x}")
    test_df["sentence"] = test_df["sentence"].str.upper()
    if cap > 0:
        test_df = test_df.iloc[:crt_i]

    print(wav2vec(test_df["path"].values, test_df["sentence"].values, "en", batch_size=batch_size))