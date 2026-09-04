# -*- coding: utf-8 -*-

import argparse
import torch
import pandas as pd
import numpy as np


def convert_pt_to_csv(input_pt, output_csv):

    # ============================
    # 读取 pt
    # ============================

    data = torch.load(input_pt, map_location="cpu")

    print("Loaded object type:", type(data))


    # ============================
    # 提取 embedding
    # ============================

    if not isinstance(data, dict):
        raise TypeError("当前pt不是预期字典格式")

    if "embeddings" not in data:
        raise ValueError("pt文件中没有 embeddings 字段")


    embeddings = data["embeddings"]

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.array(embeddings)


    names = data.get(
        "names",
        [f"sample_{i}" for i in range(len(embeddings))]
    )


    print("Embedding shape:", embeddings.shape)


    # ============================
    # 构建 dataframe
    # ============================

    df = pd.DataFrame(
        embeddings,
        columns=[f"feat_{i}" for i in range(embeddings.shape[1])]
    )

    df.insert(0, "name", names)


    # ============================
    # 保存
    # ============================

    df.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("CSV shape:", df.shape)
    print(df.head())



def parse_args():

    parser = argparse.ArgumentParser(
        description="Convert ProtT5 embedding pt file to csv"
    )

    parser.add_argument(
        "--input_pt",
        type=str,
        required=True,
        help="输入pt文件路径"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="输出csv文件路径"
    )

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    convert_pt_to_csv(
        input_pt=args.input_pt,
        output_csv=args.output_csv
    )