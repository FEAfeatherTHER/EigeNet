import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from einops import rearrange
from einops.layers.torch import Rearrange
from models.codec.codec.vocos import Vocos, VocosBackbone
from models.codec.codec.quantize import ResidualVQ
from models.codec.codec.transformers import TransformerEncoder


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=max(2 * stride, 3),
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class LatentCodecEncoder(nn.Module):
    def __init__(
        self,
        d_mel: int = 128,
        d_model: int = 128,
        num_blocks: int = 4,
        out_channels: int = 256,
        use_tanh: bool = False,
        cfg=None,
    ):
        super().__init__()

        # use cfg to set the parameters
        d_mel = cfg.d_mel if cfg is not None and hasattr(cfg, "d_mel") else d_mel
        d_model = (
            cfg.d_model if cfg is not None and hasattr(cfg, "d_model") else d_model
        )
        num_blocks = (
            cfg.num_blocks
            if cfg is not None and hasattr(cfg, "num_blocks")
            else num_blocks
        )
        out_channels = (
            cfg.out_channels
            if cfg is not None and hasattr(cfg, "out_channels")
            else out_channels
        )
        use_tanh = (
            cfg.use_tanh if cfg is not None and hasattr(cfg, "use_tanh") else use_tanh
        )

        # Create first convolution
        self.block = [WNConv1d(d_mel, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for _ in range(num_blocks):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=1)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        if use_tanh:
            self.block += [nn.Tanh()]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.reset_parameters()

    def forward(self, x, return_mel=False):
        melspec = x
        if return_mel:
            return self.block(melspec), melspec
        return self.block(melspec)

    def reset_parameters(self):
        self.apply(init_weights)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: 80,
        d_model: int = 128,
        num_blocks: int = 4,
        out_channels: int = 256,
        use_tanh: bool = False,
    ):
        super().__init__()

        in_channels = in_channels
        d_model = d_model

        # Create first convolution
        self.block = [WNConv1d(in_channels, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for _ in range(num_blocks):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=1)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        if use_tanh:
            self.block += [nn.Tanh()]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)

        self.reset_parameters()

    def forward(self, x):
        return self.block(x)

    def reset_parameters(self):
        self.apply(init_weights)


class SimpleCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: 80,
        d_model: int = 128,
        num_blocks: int = 4,
        out_channels: int = 256,
        use_tanh: bool = False,
    ):
        super().__init__()

        in_channels = in_channels
        d_model = d_model

        # Create first convolution
        self.block = [WNConv1d(in_channels, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for _ in range(num_blocks):
            d_model *= 2
            self.block += [
                nn.Sequential(
                    ResidualUnit(d_model // 2, dilation=1),
                    ResidualUnit(d_model // 2, dilation=2),
                    ResidualUnit(d_model // 2, dilation=3),
                    Snake1d(d_model // 2),
                    WNConv1d(
                        d_model // 2,
                        d_model,
                        kernel_size=max(2 * 1, 3),
                        stride=1,
                        padding=math.ceil(1 / 2),
                    ),
                )
            ]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            nn.Conv1d(d_model, out_channels, kernel_size=1),
        ]

        if use_tanh:
            self.block += [nn.Tanh()]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)

        self.reset_parameters()

    def forward(self, x):
        return self.block(x)

    def reset_parameters(self):
        self.apply(init_weights)


class LatentCodecDecoderWithTimbre(nn.Module):
    def __init__(
        self,
        d_mel: int = 128,
        in_channels: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "vq",
        quantizer_dropout: float = 0.5,
        commitment: float = 0.25,
        codebook_loss_weight: float = 1.0,
        use_l2_normlize: bool = False,
        codebook_type: str = "euclidean",
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        weight_init: bool = False,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 1152,
        vocos_num_layers: int = 8,
        ln_before_vq: bool = False,
        use_pe: bool = True,
        use_timbre_encoder: bool = True,
        cfg=None,
    ):
        super().__init__()

        # use cfg to set the parameters
        d_mel = cfg.d_mel if cfg is not None and hasattr(cfg, "d_mel") else d_mel
        in_channels = (
            cfg.in_channels
            if cfg is not None and hasattr(cfg, "in_channels")
            else in_channels
        )
        num_quantizers = (
            cfg.num_quantizers
            if cfg is not None and hasattr(cfg, "num_quantizers")
            else num_quantizers
        )
        codebook_size = (
            cfg.codebook_size
            if cfg is not None and hasattr(cfg, "codebook_size")
            else codebook_size
        )
        codebook_dim = (
            cfg.codebook_dim
            if cfg is not None and hasattr(cfg, "codebook_dim")
            else codebook_dim
        )
        quantizer_type = (
            cfg.quantizer_type
            if cfg is not None and hasattr(cfg, "quantizer_type")
            else quantizer_type
        )
        quantizer_dropout = (
            cfg.quantizer_dropout
            if cfg is not None and hasattr(cfg, "quantizer_dropout")
            else quantizer_dropout
        )
        commitment = (
            cfg.commitment
            if cfg is not None and hasattr(cfg, "commitment")
            else commitment
        )
        codebook_loss_weight = (
            cfg.codebook_loss_weight
            if cfg is not None and hasattr(cfg, "codebook_loss_weight")
            else codebook_loss_weight
        )
        use_l2_normlize = (
            cfg.use_l2_normlize
            if cfg is not None and hasattr(cfg, "use_l2_normlize")
            else use_l2_normlize
        )
        codebook_type = (
            cfg.codebook_type
            if cfg is not None and hasattr(cfg, "codebook_type")
            else codebook_type
        )
        kmeans_init = (
            cfg.kmeans_init
            if cfg is not None and hasattr(cfg, "kmeans_init")
            else kmeans_init
        )
        kmeans_iters = (
            cfg.kmeans_iters
            if cfg is not None and hasattr(cfg, "kmeans_iters")
            else kmeans_iters
        )
        decay = cfg.decay if cfg is not None and hasattr(cfg, "decay") else decay
        eps = cfg.eps if cfg is not None and hasattr(cfg, "eps") else eps
        threshold_ema_dead_code = (
            cfg.threshold_ema_dead_code
            if cfg is not None and hasattr(cfg, "threshold_ema_dead_code")
            else threshold_ema_dead_code
        )
        weight_init = (
            cfg.weight_init
            if cfg is not None and hasattr(cfg, "weight_init")
            else weight_init
        )
        vocos_dim = (
            cfg.vocos_dim
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_dim
        )
        vocos_intermediate_dim = (
            cfg.vocos_intermediate_dim
            if cfg is not None and hasattr(cfg, "vocos_intermediate_dim")
            else vocos_intermediate_dim
        )
        vocos_num_layers = (
            cfg.vocos_num_layers
            if cfg is not None and hasattr(cfg, "vocos_num_layers")
            else vocos_num_layers
        )
        ln_before_vq = (
            cfg.ln_before_vq
            if cfg is not None and hasattr(cfg, "ln_before_vq")
            else ln_before_vq
        )
        use_pe = cfg.use_pe if cfg is not None and hasattr(cfg, "use_pe") else use_pe
        use_timbre_encoder = (
            cfg.use_timbre_encoder
            if cfg is not None and hasattr(cfg, "use_timbre_encoder")
            else use_timbre_encoder
        )

        if quantizer_type == "vq":
            self.quantizer = ResidualVQ(
                input_dim=in_channels,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_type=quantizer_type,
                quantizer_dropout=quantizer_dropout,
                commitment=commitment,
                codebook_loss_weight=codebook_loss_weight,
                use_l2_normlize=use_l2_normlize,
                codebook_type=codebook_type,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                decay=decay,
                eps=eps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                weight_init=weight_init,
            )
        elif quantizer_type == "fvq":
            self.quantizer = ResidualVQ(
                input_dim=in_channels,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_type=quantizer_type,
                quantizer_dropout=quantizer_dropout,
                commitment=commitment,
                codebook_loss_weight=codebook_loss_weight,
                use_l2_normlize=use_l2_normlize,
            )
        elif quantizer_type == "lfq":
            self.quantizer = ResidualVQ(
                input_dim=in_channels,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_type=quantizer_type,
            )
        else:
            raise ValueError(f"Unknown quantizer type {quantizer_type}")

        self.model = nn.Sequential(
            VocosBackbone(
                input_channels=in_channels,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(vocos_dim, d_mel),
        )

        self.use_timbre_encoder = use_timbre_encoder
        self.ln_before_vq = ln_before_vq
        if self.use_timbre_encoder:
            self.timbre_encoder = TransformerEncoder(
                enc_emb_tokens=None,
                encoder_layer=4,
                encoder_hidden=256,
                encoder_head=4,
                conv_filter_size=1024,
                conv_kernel_size=5,
                encoder_dropout=0.1,
                use_cln=False,
                use_pe=use_pe,
                cfg=None,
            )

            self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
            self.timbre_linear.bias.data[:in_channels] = 1
            self.timbre_linear.bias.data[in_channels:] = 0
            self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)

            if self.ln_before_vq:
                self.enc_ln = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.reset_parameters()

    def forward(
        self,
        x=None,
        vq=False,
        eval_vq=False,
        n_quantizers=None,
        speaker_embedding=None,
        return_spk_embs=True,
    ):
        """
        if vq is True, x = encoder output, then return quantized output;
        else, x = quantized output, then return decoder output
        """
        if vq is True:
            if eval_vq:
                self.quantizer.eval()

            x_timbre = x

            if self.ln_before_vq and self.use_timbre_encoder:
                x = x.transpose(1, 2)
                x = self.enc_ln(x)
                x = x.transpose(1, 2)

            (
                quantized_out,
                all_indices,
                all_commit_losses,
                all_codebook_losses,
                all_quantized,
            ) = self.quantizer(x, n_quantizers=n_quantizers)

            if self.use_timbre_encoder and return_spk_embs:
                x_timbre = x_timbre.transpose(1, 2)
                x_timbre = self.timbre_encoder(x_timbre, None, None)
                x_timbre = x_timbre.transpose(1, 2)
                spk_embs = torch.mean(x_timbre, dim=2)
            else:
                spk_embs = None

            return (
                quantized_out,
                all_indices,
                all_commit_losses,
                all_codebook_losses,
                all_quantized,
                spk_embs,
            )

        if self.use_timbre_encoder:
            style = self.timbre_linear(speaker_embedding).unsqueeze(2)  # (B, 2d, 1)
            gamma, beta = style.chunk(2, 1)  # (B, d, 1)

            x = x.transpose(1, 2)
            x = self.timbre_norm(x)
            x = x.transpose(1, 2)
            x = x * gamma + beta

        return self.model(x).transpose(1, 2)

    def quantize(self, x, n_quantizers=None):
        self.quantizer.eval()
        quantized_out, vq, _, _, _ = self.quantizer(x, n_quantizers=n_quantizers)
        return quantized_out, vq

    # TODO: check consistency of vq2emb and quantize
    def vq2emb(self, vq, n_quantizers=None):
        return self.quantizer.vq2emb(vq, n_quantizers=n_quantizers)

    def decode(self, x):
        return self.model(x)

    def latent2dist(self, x, n_quantizers=None):
        return self.quantizer.latent2dist(x, n_quantizers=n_quantizers)

    def reset_parameters(self):
        self.apply(init_weights)
