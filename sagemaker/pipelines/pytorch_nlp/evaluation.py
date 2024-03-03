"""Script for Evaluation Processing Step.

Load trained model and test data file, then calculate model test metrics.
"""

import json
import logging
import pathlib
import tarfile

import numpy as np
import pandas as pd
import os

import torch
from model import TweetDataset


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


BASE_DIR = ""


def get_test_loader(df, pretrained_path, batch_size=32):
    TDS = TweetDataset(
        df, pretrained_model_path=os.path.join(pretrained_path, "tokenizer_model.pth")
    )
    loader = torch.utils.data.DataLoader(
        TDS, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return loader


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def evaluate_jaccard(eval_df: pd.DataFrame):
    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    total_jaccard = 0
    for index, row in eval_df.iterrows():
        true_selected_text = row["selected_text"]
        pred_selected_text = row["pred_selected_text"]
        score = jaccard(true_selected_text, pred_selected_text)
        total_jaccard += score
    average_jaccard = total_jaccard / len(eval_df)

    return average_jaccard


if __name__ == "__main__":
    model_folder_path = "/opt/ml/processing/model"
    model_path = f"{model_folder_path}/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=model_folder_path)

    test_path = "/opt/ml/processing/test/tune.csv"
    test_df = pd.read_csv(test_path).head(10)
    test_df["text"] = test_df["text"].astype(str)
    test_loader = get_test_loader(test_df, pretrained_path=model_folder_path)
    predictions = []
    models = []

    for fold in range(2):
        if torch.cuda.is_available():
            model = torch.load(f"{model_folder_path}/roberta_fold{fold+1}.pth")
            model.cuda()
        else:
            model = torch.load(
                f"{model_folder_path}/roberta_fold{fold+1}.pth",
                map_location=torch.device("cpu"),
            )
        model.eval()
        models.append(model)

    for data in test_loader:
        if torch.cuda.is_available():
            ids = data["ids"].cuda()
            masks = data["masks"].cuda()
        else:
            ids = torch.tensor(data["ids"])
            masks = torch.tensor(data["masks"])
        tweet = data["tweet"]
        offsets = data["offsets"].numpy()

        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_logits.append(
                    torch.softmax(output[0], dim=1).cpu().detach().numpy()
                )
                end_logits.append(
                    torch.softmax(output[1], dim=1).cpu().detach().numpy()
                )

        start_logits = np.mean(start_logits, axis=0)
        end_logits = np.mean(end_logits, axis=0)
        for i in range(len(ids)):
            start_pred = np.argmax(start_logits[i])
            end_pred = np.argmax(end_logits[i])
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)

    pred_df = pd.concat(
        [test_df, pd.Series(predictions, name="pred_selected_text")], axis=1
    )

    report_dict = {
        "metrics": {
            "jaccard": {"value": evaluate_jaccard(pred_df)},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    output_path = f"{output_dir}/pred.json"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing out test df with prediction to {output_path}")

    with open(output_path, "w") as f:
        # f.write(pred_df.to_json(orient = 'records'))
        f.write(json.dumps(report_dict))
