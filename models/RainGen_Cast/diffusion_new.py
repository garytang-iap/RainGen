import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange, repeat
from collections import namedtuple
from functools import partial
import math
from .utils_diffusion import *



ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        #distributed=True,
        *,
        timesteps = 1000,
        sampling_timesteps = 50,
        objective = 'pred_v',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.0,
        offset_noise_strength = 0.1,
        offset_noise_strategy = 'global',  # 新增：offset noise策略
        min_snr_loss_weight = False,
        min_snr_gamma = 5,# === 新增：从config.json传入的CFG相关参数 ===
        cond_drop_prob = 0.1, # 条件丢弃概率，用于训练
        use_cfg_plus_plus = False # https://arxiv.org/pdf/2406.08070
    ):
        super().__init__()
        #assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        #assert not model.random_or_learned_sinusoidal_cond

        #if distributed == True:
        #    self.model = model.module
        #else:
        #    self.model= model
         # 如果是DDP模型，就保存DDP模型；如果是普通模型，就保存普通模型。
        self.model = model
        
        # 判断是否为DDP模型，以备后用 (比如采样时)
        self.is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)
        
        # 获取未经包装的模型，用于一些非DDP操作
        self.model_raw = model.module if self.is_ddp else model
        #self.channels = self.model.channels#这个要注意，我在UNet里面设置了channels*2，这里用10就可以
        # 保存CFG参数
        self.cond_drop_prob = cond_drop_prob

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # use cfg++ when ddim sampling

        self.use_cfg_plus_plus = use_cfg_plus_plus

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training，不设置的话计算timesteps

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        # 保存offset noise相关参数
        self.offset_noise_strength = offset_noise_strength
        self.offset_noise_strategy = offset_noise_strategy
        
        # 验证offset策略参数
        valid_strategies = ['latent_wise', 'global', 'disabled']
        assert offset_noise_strategy in valid_strategies, \
            f'offset_noise_strategy must be one of {valid_strategies}, got {offset_noise_strategy}'
        

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    @property
    def device(self):
        # 从模型的参数中获取设备，这是最可靠的方式
        return next(self.model.parameters()).device
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, condition, cond_scale = 3.0, clip_x_start = False):
        """
        🔥 简化的CFG推理逻辑 - CFG逻辑现在在Latte模型内部
        """
        if cond_scale == 1.0:
            # 无引导：直接预测
            model_output = self.model_raw(x, t, condition, apply_cfg_dropout=False)
        else:
            # 有引导：需要条件和无条件两次前向传播
            batch_size = x.shape[0]
            device = x.device
            
            # 1. 有条件预测
            cond_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            cond_pred = self.model_raw(x, t, condition, 
                                     apply_cfg_dropout=False, 
                                     condition_mask=cond_mask)
            
            # 2. 无条件预测
            uncond_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            uncond_pred = self.model_raw(x, t, condition,
                                       apply_cfg_dropout=False,
                                       condition_mask=uncond_mask)
            
            # 3. CFG公式
            model_output = uncond_pred + cond_scale * (cond_pred - uncond_pred)
            
        # === 原有的预测处理逻辑 ===
        maybe_clip = partial(torch.clamp, min =-10., max = 10.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, condition, cond_scale, clip_denoised = False):
        preds = self.model_predictions(x, t, condition, cond_scale, clip_x_start=clip_denoised)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-0.38129259424209755,4.580308111519394)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    @torch.no_grad()
    def p_sample(self, x, t: int, condition, cond_scale = 6.0, clip_denoised = False):
        # 移除 rescaled_phi
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, condition = condition, cond_scale = cond_scale, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    @torch.no_grad()
    def p_sample_loop(self, condition, shape, cond_scale = 6.0):
        # 移除 rescaled_phi
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)

        for t in reversed(range(0, self.num_timesteps)):
            img, _ = self.p_sample(img, t, condition, cond_scale)

        return img

    @torch.no_grad()
    def ddim_sample(self, condition, shape, sampling_timesteps=50, ddim_sampling_eta=0.0, cond_scale = 6.0, clip_denoised = True):
        # 移除 rescaled_phi
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device)

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # 调用修正后的 model_predictions
            preds = self.model_predictions(img, time_cond, condition=condition, cond_scale=cond_scale, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.no_grad()
    def sample(self, condition, frames, sampling_timesteps = 50, ddim_sampling_eta=0.0, cond_scale = 6.0):
        # 移除了 rescaled_phi, 只保留了CFG的核心参数 cond_scale
        batch_size, _, channels, image_Height, image_width = condition.shape
    
        # 创建最终输出的 shape，格式为 BTCHW
        shape = (batch_size, frames, channels, image_Height, image_width)

        if self.is_ddim_sampling:
            return self.ddim_sample(
                condition,
                shape,
                sampling_timesteps=sampling_timesteps,
                ddim_sampling_eta=ddim_sampling_eta,
                cond_scale=cond_scale
            )
        else:
            return self.p_sample_loop(
                condition,
                shape,
                cond_scale=cond_scale
            )

    @torch.no_grad()
    def interpolate(self, x1, x2, condition, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in reversed(range(0, t)):
            img, _ = self.p_sample(img, i, condition)

        return img

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0. and self.offset_noise_strategy != 'disabled':
            b, f, h, w, c = x_start.shape
            
            if self.offset_noise_strategy == 'latent_wise':
                # 策略1：每个latent维度独立offset，时间一致（推荐用于latent diffusion）
                offset_noise = torch.randn(b, c, device=self.device)
                reshaped_offset = rearrange(offset_noise, 'b c -> b 1 c 1 1')
                noise += self.offset_noise_strength * reshaped_offset
                
            elif self.offset_noise_strategy == 'global':
                # 策略2：全局offset，所有维度一致（保守选择）
                offset_noise = torch.randn(b, device=self.device)
                reshaped_offset = rearrange(offset_noise, 'b -> b 1 1 1 1')
                noise += self.offset_noise_strength * reshaped_offset
            
            # 'disabled' 情况下不做任何处理
            
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, condition, *, noise = None):
        """🔥 简化的损失计算 - CFG dropout现在在模型内部处理"""
        b, f, h, w,c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        
        # 🔥 关键：apply_cfg_dropout=True 让Latte模型内部处理条件丢弃
        model_out = self.model(x, t, condition, apply_cfg_dropout=True)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean(),model_out.std(),target.std()

    def forward(self, img, condition, *args, **kwargs):
        """🔥 大大简化的前向传播"""
        device = self.device
        b = img.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # 🔥 不再需要在这里处理条件丢弃，Latte模型内部会处理
        return self.p_losses(img, t, condition, *args, **kwargs)