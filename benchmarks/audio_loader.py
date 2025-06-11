import torchaudio
from torch import Tensor
import numpy as np
import numpy.typing as npt
from typing import Union

class AudioLoader:
    @staticmethod
    def load(paths:npt.NDArray[np.str_], target_sampling_rate:int, as_list=False) -> Union[list[Tensor], list[list[float]]]:
        if as_list:
            return list(map(lambda x: AudioLoader.__load_path(x, target_sampling_rate).tolist(), paths))
        else:
            return list(map(lambda x: AudioLoader.__load_path(x, target_sampling_rate), paths))

    @staticmethod
    def __load_path(path:str, target_sampling_rate) -> Tensor:
        waveform, audio_sampling_rate = torchaudio.load(path)
        if audio_sampling_rate != target_sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_sampling_rate, 
                                                        new_freq=target_sampling_rate)
            waveform = resampler(waveform)
        return waveform
        

