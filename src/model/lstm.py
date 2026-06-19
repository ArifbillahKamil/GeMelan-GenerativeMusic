import torch
import torch.nn as nn


class MelodyLSTM(nn.Module):

    def __init__(
        self,
        vocab_size:  int = 256,
        embed_dim:   int = 64,
        hidden_size: int = 256,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        pad_idx:     int = 0,
    ):
        super(MelodyLSTM, self).__init__()

        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Dropout Layer
        self.dropout_layer = nn.Dropout(dropout)

        # 4. Linear (Fully Connected) Layer
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Inisialisasi bobot
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Forget gate bias = 1 (teknik umum untuk stabilitas LSTM)
                hidden_size = param.data.shape[0] // 4
                param.data[hidden_size:2*hidden_size].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # (batch, seq_len, embed_dim) -> (batch, seq_len, hidden_size)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Dropout pada output LSTM
        lstm_out = self.dropout_layer(lstm_out)

        # (batch, seq_len, hidden_size) -> (batch, seq_len, vocab_size)
        logits = self.fc(lstm_out)

        return logits, hidden

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        print("=" * 50)
        print("  GeMelan - MelodyLSTM Architecture")
        print("=" * 50)
        print(f"  Vocab size   : {self.vocab_size}")
        print(f"  Embed dim    : {self.embed_dim}")
        print(f"  Hidden size  : {self.hidden_size}")
        print(f"  Num layers   : {self.num_layers}")
        print(f"  Dropout      : {self.dropout}")
        print("-" * 50)
        print(f"  Embedding    : {self.vocab_size} x {self.embed_dim}")
        print(f"  LSTM         : {self.embed_dim} -> {self.hidden_size} (x{self.num_layers})")
        print(f"  Linear       : {self.hidden_size} -> {self.vocab_size}")
        print("-" * 50)
        print(f"  Total params : {self.count_parameters():,}")
        print("=" * 50)