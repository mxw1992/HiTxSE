# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import joblib


def predict_svm(test_csv, model_path, output_csv):

    # ============================
    # 加载未知序列
    # ============================

    data = pd.read_csv(test_csv, dtype={"name": "string"})

    names = data["name"].tolist()

    x_test = data.iloc[:, 1:].values

    print("Prediction data shape:", x_test.shape)


    # ============================
    # 加载SVM
    # ============================

    svm = joblib.load(model_path)

    print("SVM model loaded.")


    # ============================
    # 预测
    # ============================

    pred = svm.predict(x_test)

    prob = svm.predict_proba(x_test)


    # ============================
    # 整理结果
    # ============================

    results = []

    for name, label, p in zip(names, pred, prob):

        results.append({
            "name": name,
            "prediction": int(label),
            "NonSec_probability": float(p[0]),
            "Sec_probability": float(p[1])
        })


    # 打印预测结果

    for r in results:
        print(r)


    # ============================
    # 保存结果
    # ============================

    result_df = pd.DataFrame(results)

    result_df.to_csv(output_csv, index=False)

    print("Prediction finished.")
    print("Saved:", output_csv)



def parse_args():

    parser = argparse.ArgumentParser(
        description="Predict secretion proteins using SVM with ProtT5 embeddings"
    )


    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="未知序列embedding csv文件"
    )


    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的SVM模型路径(.pkl)"
    )


    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="预测结果保存路径"
    )


    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    predict_svm(
        test_csv=args.test_csv,
        model_path=args.model_path,
        output_csv=args.output_csv
    )