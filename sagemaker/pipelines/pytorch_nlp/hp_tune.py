"""Script to be executed by hyperparameter tuner Step.

Finetune NLP PyTorch model and evaluate model test results.

More info on how to construct the script can be found at
https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#prepare-a-pytorch-training-script.
"""

import argparse
import numpy as np
import pandas as pd
import os
import random
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold

import logging
import sys

from model import TweetModel, TweetDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--use-cuda", type=bool, default=False)

    # model directory
    # parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--train-data", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    # parser.add_argument("--test-data", type=str, default=os.environ["SM_CHANNEL_TEST"])

    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default=os.environ["SM_CHANNEL_PRETRAINED_DATA_DIR"],
    )

    logger.info(f"Parsed arguments: {parser.parse_known_args()}")

    return parser.parse_known_args()


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_train_val_loaders(df, train_idx, val_idx, pre_trained_dir, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, pretrained_data_dir=pre_trained_dir),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, pretrained_data_dir=pre_trained_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict


def get_test_loader(df, pretrained_path, batch_size=32):
    TDS = TweetDataset(
        df, pretrained_model_path=os.path.join(pretrained_path, "tokenizer_model.pth")
    )
    loader = torch.utils.data.DataLoader(
        TDS, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return loader


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)

    return jaccard(true, pred)


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            for data in dataloaders_dict[phase]:
                if torch.cuda.is_available():
                    ids = data["ids"].cuda()
                    masks = data["masks"].cuda()
                    tweet = data["tweet"]
                    offsets = data["offsets"].numpy()
                    start_idx = data["start_idx"].cuda()
                    end_idx = data["end_idx"].cuda()
                else:
                    ids = torch.tensor(data["ids"])
                    masks = torch.tensor(data["masks"])
                    tweet = data["tweet"]
                    offsets = data["offsets"].numpy()
                    start_idx = torch.tensor(data["start_idx"])
                    end_idx = torch.tensor(data["end_idx"])
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)

                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(ids)

                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = (
                        torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    )
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i],
                            end_logits[i],
                            offsets[i],
                        )
                        epoch_jaccard += jaccard_score

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print(
                "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard
                )
            )

    torch.save(model, filename)
    # torch.save(model.state_dict(), filename)


def evaluate_jaccard(eval_df: pd.DataFrame):
    total_jaccard = 0
    for index, row in eval_df.iterrows():
        true_selected_text = row["selected_text"]
        pred_selected_text = row["pred_selected_text"]
        score = jaccard(true_selected_text, pred_selected_text)
        total_jaccard += score
    average_jaccard = total_jaccard / len(eval_df)

    return average_jaccard


if __name__ == "__main__":
    args, unknown = parse_args()

    seed = 42
    seed_everything(seed)

    train_df = pd.read_csv(f"{args.train_data}/train.csv")
    num_epochs = args.epochs
    pre_trained_dir = args.pretrained_dir
    batch_size = 2
    n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_df, test_df = train_test_split(train_df, test_size=0.05)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df.sentiment), start=1
    ):
        print(f"Fold: {fold}")

        model = TweetModel(pre_trained_dir=args.pretrained_dir)
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
        criterion = loss_fn
        dataloaders_dict = get_train_val_loaders(
            train_df, train_idx, val_idx, pre_trained_dir, batch_size
        )

        train_model(
            model,
            dataloaders_dict,
            criterion,
            optimizer,
            num_epochs,
            f"{args.model_dir}/roberta_fold{fold}.pth",
        )

    # save tokeniser
    print("Saving pretrained_tokeniser...")
    TweetDataset(train_df, pretrained_data_dir=pre_trained_dir).save_tokenizer(
        args.model_dir
    )

    # evaluate model
    test_loader = get_test_loader(test_df.head(10), pretrained_path=args.model_dir)
    predictions = []
    models = []

    for fold in range(n_splits):
        if torch.cuda.is_available():
            model = torch.load(f"{args.model_dir}/roberta_fold{fold+1}.pth")
            model.cuda()
        else:
            model = torch.load(
                f"{args.model_dir}/roberta_fold{fold+1}.pth",
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

    pred_df = test_df.copy()
    pred_df["pred_selected_text"] = predictions
    pred_df["pred_selected_text"] = pred_df["pred_selected_text"].astype(str)

    avg_jaccard = evaluate_jaccard(pred_df)
    print(f"Test Jaccard metric: {avg_jaccard}")
