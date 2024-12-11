import torch
from torch.utils.data import DataLoader, Dataset


class NERDataset(Dataset):
    def __init__(self, data_split, tokenizer):
        self.texts = data_split["tokens"]
        self.labels = data_split["ner_tags"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        labels = self.labels[idx]

        inputs = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        labels = torch.tensor(labels + [-100] * (128 - len(labels)))[:128]
        return input_ids, attention_mask, labels
