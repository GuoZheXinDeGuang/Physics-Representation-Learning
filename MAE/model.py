from __future__ import annotations

import warnings
from functools import partial
from typing import Optional

warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="Overwriting vit_.* in registry.*", category=UserWarning)

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from timm.models.layers import trunc_normal_

from physics_jepa.videomae import (
    Block,
    VisionTransformer,
    get_sinusoid_encoding_table,
    vit_base_patch16_224,
    vit_small_patch16_224,
)


MODEL_ARCHES = {
    "pretrain_videomae_small_patch16_224": vit_small_patch16_224,
    "pretrain_videomae_base_patch16_224": vit_base_patch16_224,
}


def _build_debug_tiny_encoder(
    img_size: int,
    patch_size: int,
    in_chans: int,
    num_frames: int,
    tubelet_size: int,
) -> VisionTransformer:
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=0,
        embed_dim=48,
        depth=1,
        num_heads=3,
        mlp_ratio=2.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        all_frames=num_frames,
        tubelet_size=tubelet_size,
        use_mean_pooling=False,
    )


class MaskedAutoencoderVideo(nn.Module):
    """VideoMAE-style masked autoencoder for tensors shaped (B, C, T, H, W)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 11,
        num_frames: int = 16,
        tubelet_size: int = 2,
        encoder_arch: str = "pretrain_videomae_small_patch16_224",
        mask_ratio: float = 0.9,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        decoder_mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = bool(norm_pix_loss)

        if self.num_frames % self.tubelet_size != 0:
            raise ValueError("num_frames must be divisible by tubelet_size")
        if self.img_size % self.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        if encoder_arch == "debug_tiny":
            self.encoder = _build_debug_tiny_encoder(
                self.img_size,
                self.patch_size,
                self.in_chans,
                self.num_frames,
                self.tubelet_size,
            )
        else:
            if encoder_arch not in MODEL_ARCHES:
                raise ValueError(f"Unknown encoder_arch: {encoder_arch}")
            self.encoder = MODEL_ARCHES[encoder_arch](
                pretrained=False,
                img_size=self.img_size,
                in_chans=self.in_chans,
                all_frames=self.num_frames,
                tubelet_size=self.tubelet_size,
                num_classes=0,
                use_mean_pooling=False,
            )

        self.num_patches = self.encoder.patch_embed.num_patches
        self.num_temporal_patches = self.num_frames // self.tubelet_size
        self.num_spatial_patches = (self.img_size // self.patch_size) ** 2
        if self.num_patches != self.num_temporal_patches * self.num_spatial_patches:
            raise ValueError("Unexpected patch grid shape")

        encoder_dim = self.encoder.embed_dim
        patch_dim = self.in_chans * self.tubelet_size * self.patch_size * self.patch_size
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            "decoder_pos_embed",
            get_sinusoid_encoding_table(self.num_patches, decoder_embed_dim),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    init_values=0,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)

        self.initialize_weights()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MaskedAutoencoderVideo":
        return cls(
            img_size=int(cfg.dataset.resolution),
            patch_size=int(cfg.model.patch_size),
            in_chans=int(cfg.dataset.num_chans),
            num_frames=int(cfg.dataset.num_frames),
            tubelet_size=int(cfg.model.tubelet_size),
            encoder_arch=str(cfg.model.encoder_arch),
            mask_ratio=float(cfg.model.mask_ratio),
            decoder_embed_dim=int(cfg.model.decoder_embed_dim),
            decoder_depth=int(cfg.model.decoder_depth),
            decoder_num_heads=int(cfg.model.decoder_num_heads),
            decoder_mlp_ratio=float(cfg.model.get("decoder_mlp_ratio", 4.0)),
            norm_pix_loss=bool(cfg.model.norm_pix_loss),
        )

    def initialize_weights(self) -> None:
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def patchify(self, videos: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        t = self.tubelet_size
        return rearrange(
            videos,
            "b c (nt tt) (nh ph) (nw pw) -> b (nt nh nw) (tt ph pw c)",
            tt=t,
            ph=p,
            pw=p,
        )

    def random_tube_masking(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create VideoMAE tube masks repeated across temporal patch slices."""
        spatial_tokens = self.num_spatial_patches
        temporal_tokens = self.num_temporal_patches
        len_keep = max(1, int(spatial_tokens * (1.0 - self.mask_ratio)))

        noise = torch.rand(batch_size, spatial_tokens, device=device, generator=generator)
        spatial_ids = torch.argsort(noise, dim=1)[:, :len_keep]
        offsets = torch.arange(temporal_tokens, device=device).view(1, temporal_tokens, 1) * spatial_tokens
        ids_keep = (spatial_ids.unsqueeze(1) + offsets).reshape(batch_size, temporal_tokens * len_keep)

        mask = torch.ones(batch_size, self.num_patches, device=device)
        mask.scatter_(1, ids_keep, 0.0)
        return ids_keep, mask

    def forward_encoder(self, videos: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.patch_embed(videos)
        pos_embed = self.encoder.pos_embed.expand(tokens.shape[0], -1, -1).type_as(tokens).to(tokens.device)
        tokens = tokens + pos_embed.detach()
        tokens = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        tokens = self.encoder.pos_drop(tokens)

        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)
        return tokens

    def forward_decoder(self, encoded: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        visible_tokens = self.decoder_embed(encoded)
        batch_size = visible_tokens.shape[0]
        full_tokens = self.mask_token.type_as(visible_tokens).to(visible_tokens.device)
        full_tokens = full_tokens.expand(batch_size, self.num_patches, -1).clone()
        full_tokens.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, visible_tokens.shape[-1]), visible_tokens)
        full_tokens = full_tokens + self.decoder_pos_embed.type_as(full_tokens).to(full_tokens.device)

        for block in self.decoder_blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.decoder_norm(full_tokens)
        return self.decoder_pred(full_tokens)

    def forward_loss(self, videos: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(videos)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        loss = (pred - target).pow(2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, videos: torch.Tensor) -> dict[str, torch.Tensor]:
        ids_keep, mask = self.random_tube_masking(videos.shape[0], videos.device)
        encoded = self.forward_encoder(videos, ids_keep)
        pred = self.forward_decoder(encoded, ids_keep)
        loss = self.forward_loss(videos, pred, mask)
        return {"loss": loss, "pred": pred, "mask": mask}

    @torch.no_grad()
    def encode(self, videos: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        tokens = self.encoder.get_patch_embeddings(videos)
        if pool == "mean":
            return tokens.mean(dim=1)
        if pool == "tokens":
            return tokens
        raise ValueError(f"Unsupported pool mode: {pool}")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
