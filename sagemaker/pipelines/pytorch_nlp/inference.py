"""Script to be executed by Model Step.

Inference script to serve a model, including handling data processing, model loading, prediction and output processing.

More info on how to construct the script can be found at 
https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model.
"""

import json

import numpy as np
import pandas as pd
import os
import torch


from model import TweetDataset


JSON_CONTENT_TYPE = "application/json"
# CSV_CONTENT_TYPE = "text/csv"
BASE_DIR = ""


def model_fn(model_dir):
    """Function to load in trained models."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading the model...")
    models = []
    # Hardcoded to use 2 in the PoC.
    for fold in range(2):
        if torch.cuda.is_available():
            model = torch.load(f"{model_dir}/roberta_fold{fold+1}.pth")
            model.cuda()
        else:
            model = torch.load(
                f"{model_dir}/roberta_fold{fold+1}.pth",
                map_location=torch.device("cpu"),
            )
        model.eval()
        models.append(model)

    return models


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    """Function to handle data preprocessing before passing data to model prediction."""

    def get_test_loader(df, pretrained_path, batch_size=32):
        TDS = TweetDataset(
            df,
            pretrained_model_path=os.path.join(pretrained_path, "tokenizer_model.pth"),
        )
        loader = torch.utils.data.DataLoader(
            TDS, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return loader

    print("type(serialized_input_data): ", type(serialized_input_data))
    print("Got serialized_input_data: {}".format(serialized_input_data))

    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        test_df = pd.json_normalize(input_data)
    # elif content_type == CSV_CONTENT_TYPE:
    #     test_df = pd.read_csv(input_data, names=["text", "sentiment"])
    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return

    base_dir = ""
    test_df["text"] = test_df["text"].astype(str)

    test_loader = get_test_loader(test_df, pretrained_path=base_dir)

    return test_loader


def predict_fn(input_data, model):
    """Function to execute model predict.

    Args:
      input_data:
        Output of input_fn - preprocessed input data.
      model:
        Output of model_fn - loaded trained models.

    Returns:
      Predictions
    """

    def get_selected_text(text, start_idx, end_idx, offsets):
        selected_text = ""
        for ix in range(start_idx, end_idx + 1):
            selected_text += text[offsets[ix][0] : offsets[ix][1]]
            if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                selected_text += " "
        return selected_text

    predictions = []
    for data in input_data:
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
        for m in model:
            with torch.no_grad():
                output = m(ids, masks)
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

    print(predictions)

    return predictions


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    """Function to postprocess model prediction."""
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    # elif accept == CSV_CONTENT_TYPE:
    #     return pd.DataFrame(predictions).to_csv(), accept

    raise Exception("Requested unsupported ContentType in Accept: " + accept)
