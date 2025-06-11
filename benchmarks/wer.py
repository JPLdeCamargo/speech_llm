import jiwer
import numpy as np
import numpy.typing as npt

class WER:
    @staticmethod
    def __get_transforms():
        return jiwer.Compose([
            jiwer.ReduceToSingleSentence(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.ToLowerCase(),
            jiwer.RemoveEmptyStrings(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords()
        ])

    @staticmethod
    def get_wer(reference:list[str], transcription:list[str]) -> None:
        transforms = WER.__get_transforms()
        return jiwer.wer(
                reference,
                transcription,
                truth_transform=transforms,
                hypothesis_transform=transforms,
            )

if __name__ == "__main__":
    print(WER.get_wer(["oi"*100]*1000000, ["OI"*100]*1000000))
            