import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if labels is not None:
            valid_mask = (labels != -100) & attention_mask.bool()
            adjusted_labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
            loss = -self.crf(emissions, adjusted_labels, mask=valid_mask, reduction="mean")
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())
