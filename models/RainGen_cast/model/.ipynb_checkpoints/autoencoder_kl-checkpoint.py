from typing import Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from einops import rearrange, repeat
from .vae import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution
import numpy as np

# ConditionalEncoder 类定义（已修改）
class ConditionalEncoder(nn.Module):
    def __init__(self, vae_encoder,scale=None):
        super(ConditionalEncoder, self).__init__()
        self.vae_encoder = vae_encoder
        self.scale=scale

    def forward(self, x):
        """
        Encode input frames to latent space.
        :param x: [batch, num_frame, channels, H, W]
        :return: [batch, latent_dim, frame, H', W']
        """
        batch, num_frame, channels, H, W = x.shape
        # Reshape to [batch * num_frame, channels, H, W]
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        # Encode
        posterior = self.vae_encoder.encode(x)  # [batch * num_frame, latent_dim, H', W']
        z = posterior.mode()
        latent_dim = z.shape[1]
        # Reshape to [batch, num_frame, latent_dim, H', W']
        z = rearrange(z, '(b f) c h w -> b f c h w', b=batch, f=num_frame)
        # Permute to [batch, frame, H', W'，channel]
        z = rearrange(z, 'b f c h w -> b f h w c')
        if self.scale is not None:
            
            z=z/self.scale
        return z  # [batch, frame, H', W',c]
    

class VAEDecoder(nn.Module):
    def __init__(self, vae_decoder,scale=None):
        super(VAEDecoder, self).__init__()
        self.vae_decoder = vae_decoder
        self.scale=scale
    def forward(self, z):
        """
        Decode latent representation back to image space.
        :param z: [batch, latent_dim, frame, H, W]
        :return: [batch, frame, 1, H', W']
        """
        batch,  num_frames, H, W,Latent_dim = z.shape
        
        # Reshape to [batch * frame, latent_dim, H, W]
        if self.scale is not None:
            z=z*self.scale
        
        z = rearrange(z, 'b f h w c-> (b f) c h w')
        
        # Decode
        x = self.vae_decoder.decode(z)  # [batch * frame, channels, H', W']
        
        # Reshape to [batch, frame, channels, H', W']
        x = rearrange(x, '(b f) c h w -> b f  h w c', b=batch, f=num_frames)
        
        return x  # [batch, frame, H', W',channels]





class AutoencoderKL(nn.Module):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        down_block_types: Tuple[str] = ('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
        up_block_types: Tuple[str] = ('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 64,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.use_slicing = False

    def encode(self, x: torch.FloatTensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def _decode(self, z: torch.FloatTensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously invoked, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def decode(self, z: torch.FloatTensor) -> torch.Tensor:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)
        return decoded

    def forward(
            self,
            sample: torch.FloatTensor,
            sample_posterior: bool = False,
            return_posterior: bool = True,
            generator: Optional[torch.Generator] = None,
        ) -> torch.FloatTensor:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_posterior (`bool`, *optional*, defaults to `False`):
                Whether or not to return `posterior` along with `dec` for calculating the training loss.
        """
        x = sample
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if return_posterior:
            return dec, posterior
        else:
            return dec
if __name__ == "__main__":
    
    # Instantiate the VAE
    model = AutoencoderKL(in_channels=1,latent_channels=64)

    # Dummy Input
    x = torch.randn(1,1,128, 128)  # Example with 25x256 RGB image

    # Forward Pass
    decoded, encoded = model(sample=x)
    print(encoded.sample().shape)
    def count_parameters(model, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    print(count_parameters(model,only_trainable=True))
    print(encoded.mode().shape)