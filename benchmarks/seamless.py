from transformers import SeamlessM4TModel, AutoProcessor
import torch
import torchaudio
import numpy.typing as npt
import numpy as np
import pandas as pd
from typing import Literal
from datasets import Dataset

from wer import WER
from audio_loader import AudioLoader


def seamless(paths:npt.NDArray[np.str_], reference:npt.NDArray[np.str_], language:Literal['en', 'pt'], batch_size:int):
    language_map = {"en":"eng", "pt":"por"}
    if not language in language_map:
        raise TypeError("language not found") 

    # Load the pre-trained SeamlessM4T model from the 🤗 Transformers Hub
    # "facebook/seamless-m4t-v2-large"
    model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-v2-large")

    # Check if CUDA is available, if yes, set the device to "cuda:0", else use the CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    # Move the model to the specified device (CUDA if available, otherwise CPU)
    model = model.to(device)

    # Load the pre-trained SeamlessM4T medium checkpoint using the AutoProcessor
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

    # Extracting the sample rate from the model's configuration
    sample_rate = model.config.sampling_rate
    print("model loaded")
    dataset = Dataset.from_dict({"paths":paths, "reference":reference})

    def handle_batch(batch):
        # Load the audio file
        paths = batch["paths"]
        reference_batch = batch["reference"]
        audio_sample = AudioLoader.load(paths, 16000)

        # Process the audio inputs using the specified processor, device, and sampling rate
        audio_inputs = processor(audios=audio_sample, return_tensors="pt", sampling_rate=sample_rate).to(device)

        # Generate text from the processed audio inputs, targeting French as the output language and disabling speech generation
        output_tokens = model.generate(**audio_inputs, tgt_lang=language_map[language], generate_speech=False).sequences

        # Decode the output tokens to obtain the translated text from the audio
        translated_text_from_audio = processor.batch_decode(output_tokens, skip_special_tokens=True)
        print(translated_text_from_audio)
        print(reference_batch)
        return {"wer":[WER.get_wer(reference=list(reference_batch), transcription=list(translated_text_from_audio))]}

    wers = dataset.map(handle_batch, batched=True, batch_size=batch_size, remove_columns=["paths", "reference"])
    return wers.with_format("pandas")["wer"].mean()

if __name__ == "__main__":
    batch_size = int(input("batch_size: "))
    cap = int(input("Total tests: "))
    if cap > 0:
        crt_i = cap
    test_df = pd.read_csv('librispeech.csv')
    test_df["sentence"] = test_df["sentence"].str.upper()
    # test_df = pd.read_csv('data/cv-corpus-19.0-2024-09-13/pt/test.tsv', sep='\t')
    # test_df["path"] = test_df["path"].apply(lambda x: f"data/cv-corpus-19.0-2024-09-13/pt/clips/{x}")
    # test_df["sentence"] = test_df["sentence"].str.upper()
    if cap > 0:
        test_df = test_df.iloc[:crt_i]

    print(seamless(test_df["path"].values, test_df["sentence"].values, "en", batch_size=batch_size))