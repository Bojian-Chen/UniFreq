import torch
import torch.nn as nn


class RandomPatchEmbedding(nn.Module):
    def __init__(self, patch_len, num_patches, d_model, embedding_fft="local"):
        super().__init__()
        self.patch_len = int(patch_len)
        self.num_patches = int(num_patches)
        self.embedding_fft = embedding_fft
        self.projector = nn.Sequential(
            nn.Linear(self.patch_len, int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )

    def _patch(self, x):
        batch_size, num_channels, seq_len = x.shape
        if self.embedding_fft == "global":
            x = torch.fft.fft(x, dim=-1).abs()

        if self.patch_len > seq_len:
            repeats = (self.patch_len + seq_len - 1) // seq_len
            x = x.repeat(1, 1, repeats)
            seq_len = int(x.shape[-1])

        max_start = seq_len - self.patch_len
        starts = torch.randint(
            0,
            max_start + 1,
            (batch_size, self.num_patches),
            device=x.device,
        )
        offsets = torch.arange(self.patch_len, device=x.device)
        indices = starts.unsqueeze(-1) + offsets
        indices = indices.unsqueeze(2)

        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(3, indices.expand(-1, -1, num_channels, -1))
        patches = patches.permute(0, 2, 1, 3).contiguous()

        if self.embedding_fft == "local":
            patches = torch.fft.fft(patches, dim=-1).abs()
        return patches

    def forward(self, x):
        return self.projector(self._patch(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ffn, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ffn)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ffn), int(d_model)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TFProFMBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_classes = int(args.num_classes)
        self.input_len = int(args.tf_profm_input_len)
        self.patch_len = int(args.tf_profm_patch_len)
        self.num_patches = int(args.tf_profm_num_patches)
        self.d_model = int(args.tf_profm_d_model)
        self.d_ffn = int(args.tf_profm_d_ffn)
        self.feature_dim = int(args.tf_profm_feature_dim)

        d_enc_out = self.num_patches * self.d_model
        self.embedding = RandomPatchEmbedding(
            patch_len=self.patch_len,
            num_patches=self.num_patches,
            d_model=self.d_model,
            embedding_fft="local",
        )
        self.encoder = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                d_ffn=self.d_ffn,
                heads=int(args.tf_profm_heads),
                dropout=float(args.dropout),
            )
            for _ in range(int(args.tf_profm_layers))
        ])
        self.feature_projector = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(d_enc_out, self.feature_dim),
            nn.ReLU(),
        )

        if self.num_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, int(args.tf_profm_head_hidden)),
                nn.ReLU(),
                nn.Linear(int(args.tf_profm_head_hidden), self.num_classes),
            )
        else:
            self.head = None

        self._finetuning_enabled = False
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):
        return super().train(bool(mode) if self._finetuning_enabled else False)

    def enable_finetuning(self):
        """Match the repository's explicit downstream fine-tuning contract."""
        self._finetuning_enabled = True
        self.requires_grad_(True)
        return super().train(True)

    def disable_finetuning(self):
        self._finetuning_enabled = False
        self.requires_grad_(False)
        return super().train(False)

    def _encode_channels(self, x):
        if x.ndim != 3:
            raise ValueError("TF-ProFM expects input shape [B, C, L], got %s" % (tuple(x.shape),))
        if int(x.shape[-1]) != self.input_len:
            raise ValueError(
                "TF-ProFM input length is %d, but model was built for %d"
                % (int(x.shape[-1]), self.input_len)
            )

        tokens = self.embedding(x)
        batch_size, num_channels, _, _ = tokens.shape
        tokens = tokens.view(batch_size * num_channels, self.num_patches, self.d_model)
        for block in self.encoder:
            tokens = block(tokens)
        features = self.feature_projector(tokens)
        features = features.view(batch_size, num_channels, self.feature_dim)
        return features

    def forward(self, x):
        if isinstance(x, dict):
            x = x["time_x"]
        return self._encode_channels(x).mean(dim=1)

    def predict(self, features):
        if self.head is None:
            return features
        return self.head(features)

    def classify(self, x):
        return self.predict(self.forward(x))

