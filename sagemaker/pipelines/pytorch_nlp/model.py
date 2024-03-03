"""Model code."""

import os
import tokenizers
import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig


class TweetModel(nn.Module):
    def __init__(self, pre_trained_dir="", inference_mode=False):
        super(TweetModel, self).__init__()

        if not inference_mode:
            config = RobertaConfig.from_pretrained(
                os.path.join(pre_trained_dir, "config.json"), output_hidden_states=True
            )
            self.roberta = RobertaModel.from_pretrained(
                os.path.join(pre_trained_dir, "pytorch_model.bin"), config=config
            )
        else:
            config = RobertaConfig()
            self.roberta = RobertaModel(config=config)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
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


class TweetDataset(torch.utils.data.Dataset):
    def __init__(
        self, df, pretrained_data_dir=None, pretrained_model_path=None, max_len=96
    ):
        self.df = df
        self.max_len = max_len
        self.labeled = "selected_text" in df
        if pretrained_data_dir:
            self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                vocab=os.path.join(pretrained_data_dir, "vocab.json"),
                merges=os.path.join(pretrained_data_dir, "merges.txt"),
                lowercase=True,
                add_prefix_space=True,
            )
        else:
            self.tokenizer = torch.load(pretrained_model_path)

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

    def save_tokenizer(self, save_path):
        torch.save(self.tokenizer, f"{save_path}/tokenizer_model.pth")
