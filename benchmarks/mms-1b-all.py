from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import Dataset
import pandas as pd
import torch
import numpy.typing as npt
import numpy as np
from typing import Literal
from wer import WER
import torchaudio
from audio_loader import AudioLoader

# crt_i = 2
# test_df = pd.read_csv('librispeech.csv')
# test_df["sentence"] = test_df["sentence"].str.upper()
# test_df = test_df.iloc[:crt_i]
# # test_df = pd.read_csv('data/cv-corpus-19.0-2024-09-13/pt/test.tsv', sep='\t')
# # test_df["path"] = test_df["path"].apply(lambda x: f"data/cv-corpus-19.0-2024-09-13/pt/clips/{x}")
# # test_df = test_df.iloc[:crt_i]

# waveform, sample_rate = torchaudio.load(test_df["path"][1])
# waveform = waveform.mean(dim=0)


# # sound_array = sf.read(test_df["path"][0])[0]
# # sound_array = sound_array.tolist()

# # PT
# import torch



# inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs).logits

# ids = torch.argmax(outputs, dim=-1)[0]
# transcription = processor.decode(ids)
# print(transcription)

def mms_1b_all(paths:npt.NDArray[np.str_], reference:npt.NDArray[np.str_], language:Literal['en', 'pt'], batch_size:int):
    language_map = {"en":"eng", "pt":"por"}
    if not language in language_map:
        raise TypeError("language not found") 

    model_id = "facebook/mms-1b-all"

    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    processor.tokenizer.set_target_lang(language_map[language])
    model.load_adapter(language_map[language])
        
    dataset = Dataset.from_dict({"paths":paths, "reference":reference})

    def handle_batch(batch):
        audio_array = AudioLoader.load(batch["paths"], 16000, as_list=True)
        audio_array = [np.squeeze(audio) for audio in audio_array]
        inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt", batch_size=batch_size, padding=True).input_values

        with torch.no_grad():
            outputs = model(inputs).logits

        ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(ids)
        return {"wer":[WER.get_wer(reference=list(batch["reference"]), transcription=list(transcription))]}


    wers = dataset.map(handle_batch, batched=True, batch_size=batch_size, remove_columns=["paths", "reference"])
    return wers.with_format("pandas")["wer"].mean()



if __name__ == "__main__":
    crt_i = 2
    # test_df = pd.read_csv('librispeech.csv')
    # test_df["sentence"] = test_df["sentence"].str.upper()
    # test_df = test_df.iloc[:crt_i]

    test_df = pd.read_csv('data/cv-corpus-19.0-2024-09-13/pt/test.tsv', sep='\t')
    test_df["path"] = test_df["path"].apply(lambda x: f"data/cv-corpus-19.0-2024-09-13/pt/clips/{x}")
    test_df["sentence"] = test_df["sentence"].str.upper()
    test_df = test_df.iloc[:crt_i]


    print(mms_1b_all(test_df["path"].values, test_df["sentence"].values, "pt", batch_size=2))