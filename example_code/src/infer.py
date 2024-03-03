import numpy as np
import pandas as pd
import os
import warnings
import random
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig


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


seed = 42
seed_everything(seed)


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.max_len = max_len
        self.labeled = "selected_text" in df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab="../data/roberta_base/vocab.json",
            merges="../data/roberta_base/merges.txt",
            lowercase=True,
            add_prefix_space=True,
        )

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet
        data["offsets"] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]

        return start_idx, end_idx


def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=2
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict


def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return loader


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        config = RobertaConfig.from_pretrained(
            "../data/roberta_base/config.json", output_hidden_states=True
        )
        self.roberta = RobertaModel.from_pretrained(
            "../data/roberta_base/pytorch_model.bin", config=config
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)
        computed = self.roberta(input_ids, attention_mask)
        hs = computed[2]

        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


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
    start_pred = np.argmax(
        start_logits
    )  # starting point of substring for the predicted sentiment
    end_pred = np.argmax(
        end_logits
    )  # ending point of substring for the predicted sentiment
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

    torch.save(model.state_dict(), filename)


num_epochs = 1
batch_size = 2
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# TODO 1: Sepearate training file into a tuning file and a training file
def make_tuning_data_file():
    from sklearn.model_selection import train_test_split

    source_filename = "../data/kaggle/tweet-sentiment-extraction/train.csv"
    train_df = pd.read_csv(source_filename)
    train, tune = train_test_split(train_df, test_size=0.05, random_state=100)
    train.to_csv("../data/kaggle/tweet-sentiment-extraction/train.csv")
    tune.to_csv("../data/kaggle/tweet-sentiment-extraction/tune.csv")


# TODO 2: Write an inference function
def get_predicted_substrings():
    tune_df = pd.read_csv("../data/kaggle/tweet-sentiment-extraction/tune.csv")
    tune_df["text"] = tune_df["text"].astype(str)
    tune_loader = get_test_loader(tune_df)
    predictions = []
    models = []
    for fold in range(skf.n_splits):
        model = TweetModel()
        if torch.cuda.is_available():
            model.cuda()
            model.load_state_dict(
                torch.load(f"../data/model_trained/roberta_fold{fold+1}.pth")
            )
        else:
            model.load_state_dict(
                torch.load(
                    f"../data/model_trained/roberta_fold{fold+1}.pth",
                    map_location=torch.device("cpu"),
                )
            )
        model.eval()
        models.append(model)

    for data in tune_loader:
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

    tune_df["pred_selected_text"] = predictions
    return tune_df


# TODO 3: Use Inference function to get true and predicted "selected_text" values and evaluate jaccard on them
def evaluate_jaccard(eval_df: pd.DataFrame):
    total_jaccard = 0
    for index, row in eval_df.iterrows():
        true_selected_text = row["selected_text"]
        pred_selected_text = row["pred_selected_text"]
        score = jaccard(true_selected_text, pred_selected_text)
        total_jaccard += score
    average_jaccard = total_jaccard / len(eval_df)

    return average_jaccard


def write_inference_eval_to_disk():
    infered_df = get_predicted_substrings()
    score = evaluate_jaccard(eval_df=infered_df)
    with open("inference_evaluation.txt", "w") as file:
        msg = "Average Jaccard Score: " + str(score)
        file.write(msg)


make_tuning_data_file()
write_inference_eval_to_disk()
