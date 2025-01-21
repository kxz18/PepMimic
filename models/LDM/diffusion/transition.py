import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_transition(_type, num_steps, opt):
    if _type == 'Diffusion':
        return ContinuousTransition(num_steps, opt)
    elif _type == 'FlowMatching':
        return FlowMatchingTransition(num_steps, opt)
    else:
        raise NotImplementedError(f'transition type {_type} not implemented')


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class ContinuousTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def get_timestamp(self, t):
        # use beta as timestamp
        return self.var_sched.betas[t]

    def add_noise(self, p_0, mask_generate, batch_ids, t):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha_bar = self.var_sched.alpha_bars[t] # [batch_size]
        alpha_bar = alpha_bar[batch_ids]  # [N]

        c0 = torch.sqrt(alpha_bar).view(*expand_shape)
        c1 = torch.sqrt(1 - alpha_bar).view(*expand_shape)

        e_rand = torch.randn_like(p_0)  # [N, 14, 3]
        supervise_e_rand = e_rand.clone()
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, supervise_e_rand

    def q_posterior_mean_variance(self, p_0, p_t, batch_ids, t):
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]

        betas = self.var_sched.betas
        alpha_bars = self.var_sched.alpha_bars
        alpha_bars_prev = torch.cat([alpha_bars[:-1], torch.ones_like(betas)[:1]])
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        print(posterior_variance)
        print(self.var_sched.sigmas)
        print()
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=-1)
        )

        betas = betas[t] # batch size
        alphas = self.var_sched.alphas[t] # batch size
        alpha_bars_prev = alpha_bars_prev[t] # batch_size
        alpha_bars = alpha_bars[t]
        posterior_variance = posterior_variance[t]
        posterior_log_variance_clipped = posterior_log_variance_clipped[t]

        betas = betas[batch_ids] # N
        alphas = alphas[batch_ids]
        alpha_bars_prev = alpha_bars_prev[batch_ids]
        alpha_bars = alpha_bars[batch_ids]
        posterior_variance = posterior_variance[batch_ids]
        posterior_log_variance_clipped = posterior_log_variance_clipped[batch_ids]

        c0 = (betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)).view(*expand_shape)
        c1 = ((1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)).view(*expand_shape)
        posterior_mean = c0 * p_0 + c1 * p_t

        posterior_variance = posterior_variance.view(*expand_shape).expand_as(p_t)
        posterior_log_variance_clipped = posterior_log_variance_clipped.view(*expand_shape).expand_as(p_t)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, p_t, eps_p, batch_ids, t):
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]

        sqrt_recip_alpha_bars = torch.sqrt(1.0 / self.var_sched.alpha_bars[t]) # batch_size
        sqrt_recipm1_alpha_bars = torch.sqrt(1.0 / self.var_sched.alpha_bars[t] - 1.0)  # batch_size

        sqrt_recip_alpha_bars = sqrt_recip_alpha_bars[batch_ids] # N
        sqrt_recipm1_alpha_bars = sqrt_recipm1_alpha_bars[batch_ids]

        p_0_pred = sqrt_recip_alpha_bars.view(*expand_shape) * p_t - sqrt_recipm1_alpha_bars.view(*expand_shape) * eps_p
        model_mean, _, _ = self.q_posterior_mean_variance(p_0_pred, p_t, batch_ids, t)

        # we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        model_variance = torch.cat([
            self.var_sched.betas[1:2],
            self.var_sched.betas[1:]], dim=-1)[t] # batch_size
        # model_variance = self.var_sched.sigmas[t]
        model_variance = model_variance[batch_ids].view(*expand_shape).expand_as(p_t)
        model_log_variance = torch.log(model_variance)

        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': p_0_pred,
        }

    def denoise(self, p_t, eps_p, mask_generate, batch_ids, t, guidance=None, guidance_weight=1.0):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )[batch_ids]
        alpha_bar = self.var_sched.alpha_bars[t][batch_ids]
        sigma = self.var_sched.sigmas[t][batch_ids].view(*expand_shape)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(*expand_shape)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(*expand_shape)

        z = torch.where(
            (t > 1).view(*expand_shape).expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        if guidance is not None:
            eps_p = eps_p - torch.sqrt(1 - alpha_bar).view(*expand_shape) * guidance

        # if guidance is not None:
        #     p_next = c0 * (p_t - c1 * eps_p) + sigma * z + sigma * sigma * guidance_weight * guidance
        # else:
        #     p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next


# TODO: flow matching (uniform or OT), not done yet
class FlowMatchingTransition(nn.Module):

    def __init__(self, num_steps, opt={}):
        super().__init__()
        self.num_steps = num_steps
        # TODO: number of steps T or T + 1
        c1 = torch.arange(0, num_steps + 1).float() / num_steps
        c0 = 1 - c1
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)

    def get_timestamp(self, t):
        # use c1 as timestamp
        return self.c1[t]

    def add_noise(self, p_0, mask_generate, batch_ids, t):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        c0 = self.c0[t][batch_ids].view(*expand_shape)
        c1 = self.c1[t][batch_ids].view(*expand_shape)

        e_rand = torch.randn_like(p_0)  # [N, 14, 3]
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, (e_rand - p_0)

    def denoise(self, p_t, eps_p, mask_generate, batch_ids, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        p_next = p_t - eps_p / self.num_steps
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next
