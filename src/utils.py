import json
import pandas as pd


def read_csv(file_path):
    df = pd.read_csv(file_path)
    data = []
    for i in range(df.shape[0]):
        data.append(dict(df.iloc[i]))
    return data


def save_csv(data, save_path):
    df = pd.DataFrame(data)
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        df.to_csv(f, index=False, header=f.tell()==0)


def json2dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def eval_router(data, all=240, num_models=15):
    acc = 0
    for i in range(all):
        c = 0
        m = 0
        for j in range(num_models):
            if data[i * num_models + j]['pred'] > m:
                c = data[i * num_models + j]['label']
                m = data[i * num_models + j]['pred']
        acc += c
    return acc / all
