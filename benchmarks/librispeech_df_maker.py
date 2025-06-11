import pandas as pd
import os



def recursive_folder_walker(crt_path, dfs):
    content_names = os.listdir(crt_path)
    for content in content_names:
        if '.' in content:
            splitted = content.split('.')
            if splitted[1] == 'trans':
                trans_csv = pd.read_csv(f"{crt_path}/{content}", usecols=[0], names=["temp"])
                trans_csv[["id", "sentence"]] = trans_csv['temp'].str.split(' ', n=1, expand=True)
                trans_csv["path"] = trans_csv["id"].apply(lambda x: f"{crt_path}/{x}.flac")
                trans_csv.drop(["temp"], axis=1, inplace=True)
                dfs.append(trans_csv)
        else:
            recursive_folder_walker(f"{crt_path}/{content}", dfs)

dfs = []
recursive_folder_walker("data/LibriSpeech/test-other", dfs)
res_csv = pd.concat(dfs)
res_csv.to_csv("librispeech.csv")
print(res_csv.head())