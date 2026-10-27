import torch
import torch.nn as nn

from .unifault_attention import AttentionLayer, Encoder, EncoderLayer, FullAttention


class PatchEmbed(nn.Module):
    """Split each channel into patches and project patches to embeddings."""

    def __init__(self, seq_len, patch_size=16, stride=16, embed_dim=128):
        super().__init__()
        if stride != patch_size:
            raise ValueError("This simple implementation expects stride == patch_size")
        if seq_len % patch_size != 0:
            raise ValueError(
                "seq_len (%d) must be divisible by patch_size (%d)"
                % (seq_len, patch_size)
            )

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = seq_len // patch_size
        self.input_layer = nn.Linear(patch_size, embed_dim)

    def patchify(self, x):
        if x.ndim != 3:
            raise ValueError("Expected input shape [B, C, L], got %s"
                             % (tuple(x.shape),))
        if x.shape[-1] != self.seq_len:
            raise ValueError(
                "Input sequence length is %d, but model was built for %d"
                % (x.shape[-1], self.seq_len)
            )

        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        patches = patches.contiguous()
        batch_size, num_channels, num_patches, patch_size = patches.shape
        return patches.view(batch_size * num_channels, num_patches, patch_size)

    def forward(self, x, time_mask=None, channel_count=None):
        patches = self.patchify(x)
        return self.input_layer(patches)


class UniFaultBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_channels = int(args.num_channels)
        self.embed_dim = int(args.embed_dim)
        self.num_classes = int(args.num_classes)

        self.patch_embed = PatchEmbed(
            seq_len=int(args.seq_len),
            patch_size=int(args.unifault_patch_size),
            stride=int(args.unifault_patch_size),
            embed_dim=self.embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=float(args.dropout))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            1,
                            attention_dropout=float(args.dropout),
                            output_attention=False,
                        ),
                        self.embed_dim,
                        int(args.heads),
                    ),
                    self.embed_dim,
                    4 * self.embed_dim,
                    dropout=float(args.dropout),
                    activation="gelu",
                )
                for _ in range(int(args.depth))
            ],
            norm_layer=nn.LayerNorm(self.embed_dim),
        )

        if self.num_classes > 0:
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else:
            self.head = None
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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

    def forward(self, x):
        time_mask = None
        channel_count = None
        if isinstance(x, dict):
            time_mask = x.get("time_mask")
            channel_count = x.get("channel_count")
            x = x["time_x"]
        if x.ndim != 3 or x.shape[1] != self.num_channels:
            raise ValueError("UniFault expects [B, num_channels, seq_len]")
        batch_size = x.shape[0]
        x_patch = self.patch_embed(
            x,
            time_mask=time_mask,
            channel_count=channel_count,
        )
        x_patch = x_patch + self.pos_embed
        x_patch = self.pos_drop(x_patch)

        features, _ = self.encoder(x_patch)
        features = features.reshape(
            batch_size,
            self.num_channels * self.patch_embed.num_patches,
            self.embed_dim,
        )
        return features.mean(dim=1)

    def predict(self, features):
        features_flat = features
        if self.head is None:
            return features_flat
        return self.head(features_flat)

    def classify(self, x):
        return self.predict(self.forward(x))

