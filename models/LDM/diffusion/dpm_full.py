import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
import numpy as np

from torch.autograd import grad
from torch_scatter import scatter_mean

from utils.nn_utils import variadic_meshgrid

from .transition import construct_transition
from .vlb import normal_kl, mean_flat, discretized_gaussian_log_likelihood

from ...dyMEAN.modules.am_egnn import AMEGNN
from ...dyMEAN.modules.radial_basis import RadialBasis


def low_trianguler_inv(L):
    # L: [bs, 3, 3]
    L_inv = torch.linalg.solve_triangular(L, torch.eye(3).unsqueeze(0).expand_as(L).to(L.device), upper=False)
    return L_inv


def vb_coefficient(transition, t):
    beta_t = transition.var_sched.betas[t]
    sigma_t = transition.var_sched.sigmas[t]
    alpha_t = transition.var_sched.alphas[t]
    m1_alpha_bar_t = 1.0 - transition.var_sched.alpha_bars[t]
    return 0.5 * beta_t ** 2 / (sigma_t ** 2 * alpha_t * m1_alpha_bar_t)


class EpsilonNet(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            n_channel,
            n_layers=3,
            edge_size = 0,
            n_rbf=0,
            cutoff=1.0,
            dropout=0.1,
            additional_pos_embed=True
        ):
        super().__init__()
        
        atom_embed_size = hidden_size // 4
        edge_embed_size = hidden_size // 4
        pos_embed_size, seg_embed_size = input_size, input_size
        # enc_input_size = input_size + seg_embed_size + 3 + (pos_embed_size if additional_pos_embed else 0)
        enc_input_size = input_size + 3 + (pos_embed_size if additional_pos_embed else 0)
        self.encoder = AMEGNN(
            enc_input_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=edge_embed_size + edge_size, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
        self.hidden2input = nn.Linear(hidden_size, input_size)
        # self.pos_embed2latent = nn.Linear(hidden_size, pos_embed_size)
        # self.segment_embedding = nn.Embedding(2, seg_embed_size)
        self.edge_embedding = nn.Embedding(2, edge_embed_size)

    def forward(
            self, H_noisy, X_noisy, position_embedding, ctx_edges, inter_edges,
            atom_embeddings, atom_weights, mask_generate, beta,
            ctx_edge_attr=None, inter_edge_attr=None
        ):
        """
        Args:
            H_noisy: (N, hidden_size)
            X_noisy: (N, 14, 3)
            mask_generate: (N)
            batch_ids: (N)
            beta: (N)
        Returns:
            eps_H: (N, hidden_size)
            eps_X: (N, 14, 3)
        """
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        # seg_embed = self.segment_embedding(mask_generate.long())
        if position_embedding is None:
            # in_feat = torch.cat([H_noisy, t_embed, seg_embed], dim=-1) # [N, hidden_size * 2 + 3]
            in_feat = torch.cat([H_noisy, t_embed], dim=-1) # [N, hidden_size * 2 + 3]
        else:
            # in_feat = torch.cat([H_noisy, t_embed, self.pos_embed2latent(position_embedding), seg_embed], dim=-1) # [N, hidden_size * 3 + 3]
            in_feat = torch.cat([H_noisy, t_embed, position_embedding], dim=-1) # [N, hidden_size * 3 + 3]
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        edge_embed = torch.cat([
            torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0]) # [E]
        ], dim=-1)
        edge_embed = self.edge_embedding(edge_embed) # [E, embed size]
        if ctx_edge_attr is None:
            edge_attr = edge_embed
        else:
            edge_attr = torch.cat([
                edge_embed,
                torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)
            ], dim=-1) # [E, embed size + edge_attr_size]
        next_H, next_X = self.encoder(in_feat, X_noisy, edges, ctx_edge_attr=edge_attr, channel_attr=atom_embeddings, channel_weights=atom_weights)

        # equivariant vector features changes
        eps_X = next_X - X_noisy
        eps_X = torch.where(mask_generate[:, None, None].expand_as(eps_X), eps_X, torch.zeros_like(eps_X)) 

        # invariant scalar features changes
        next_H = self.hidden2input(next_H)
        eps_H = next_H - H_noisy
        eps_H = torch.where(mask_generate[:, None].expand_as(eps_H), eps_H, torch.zeros_like(eps_H))

        return eps_H, eps_X


class FullDPM(nn.Module):

    def __init__(
        self, 
        latent_size,
        hidden_size,
        n_channel,
        num_steps, 
        n_layers=3,
        dropout=0.1,
        trans_pos_type='Diffusion',
        trans_seq_type='Diffusion',
        trans_pos_opt={}, 
        trans_seq_opt={},
        n_rbf=0,
        cutoff=1.0,
        std = 10.0,
        additional_pos_embed=True,
        dist_rbf=0,
    ):
        super().__init__()
        self.eps_net = EpsilonNet(
            latent_size, hidden_size, n_channel, n_layers=n_layers, edge_size=dist_rbf,
            n_rbf=n_rbf, cutoff=cutoff, dropout=dropout, additional_pos_embed=additional_pos_embed
        )
        if dist_rbf > 0:
            self.dist_rbf = RadialBasis(dist_rbf, 10.0)
        self.num_steps = num_steps
        self.trans_x = construct_transition(trans_pos_type, num_steps, trans_pos_opt)
        self.trans_h = construct_transition(trans_seq_type, num_steps, trans_seq_opt)

        self.register_buffer('std', torch.tensor(std, dtype=torch.float))

    def _normalize_position(self, X, batch_ids, mask_generate, atom_mask, L=None):
        ctx_mask = (~mask_generate[:, None].expand_as(atom_mask)) & atom_mask
        centers = scatter_mean(X[ctx_mask], batch_ids[:, None].expand_as(ctx_mask)[ctx_mask], dim=0) # [bs, 3]
        # print(centers[0] * 1000)
        centers = centers[batch_ids].unsqueeze(1) # [N, 1, 3]
        if L is None:
            X = (X - centers) / self.std
        else:
            with torch.no_grad():
                L_inv = low_trianguler_inv(L)
                # print(L_inv[0])
            X = X - centers
            X = torch.matmul(L_inv[batch_ids][..., None, :, :], X.unsqueeze(-1)).squeeze(-1)
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids, L=None):
        if L is None:
            X = X_norm * self.std + centers
        else:
            X = torch.matmul(L[batch_ids][..., None, :, :], X_norm.unsqueeze(-1)).squeeze(-1) + centers
        return X
    
    @torch.no_grad()
    def _get_batch_ids(self, mask_generate, lengths):

        # batch ids
        batch_ids = torch.zeros_like(mask_generate).long()
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)

        return batch_ids

    @torch.no_grad()
    def _get_edges(self, mask_generate, batch_ids, lengths):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)
        
        is_ctx = mask_generate[row] == mask_generate[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]
        return ctx_edges, inter_edges

    @torch.no_grad()
    def _get_edge_dist(self, X, edges, atom_mask):
        '''
        Args:
            X: [N, 14, 3]
            edges: [2, E]
            atom_mask: [N, 14]
        '''
        ca_x = X[:, 1] # [N, 3]
        no_ca_mask = torch.logical_not(atom_mask[:, 1]) # [N]
        ca_x[no_ca_mask] = X[:, 0][no_ca_mask] # latent coordinates
        dist = torch.norm(ca_x[edges[0]] - ca_x[edges[1]], dim=-1)  # [N]
        return dist


    def forward(
            self, H_0, X_0, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask,
        L=None, t=None, sample_structure=True, sample_sequence=True, return_states=False, batch_reduction=True):
        # if L is not None:
        #     L = L / self.std
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        batch_size = batch_ids.max() + 1
        if t == None:
            # t = torch.randint(0, self.num_steps, (batch_size,), dtype=torch.long, device=H_0.device)
            # t = torch.randint(1, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)
            t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)
        X_0, centers = self._normalize_position(X_0, batch_ids, mask_generate, atom_mask, L)

        if sample_structure:
            X_noisy, eps_X = self.trans_x.add_noise(X_0, mask_generate, batch_ids, t)
        else:
            X_noisy, eps_X = X_0, torch.zeros_like(X_0)
        if sample_sequence:
            H_noisy, eps_H = self.trans_h.add_noise(H_0, mask_generate, batch_ids, t)
        else:
            H_noisy, eps_H = H_0, torch.zeros_like(H_0)

        ctx_edges, inter_edges = self._get_edges(mask_generate, batch_ids, lengths)
        if hasattr(self, 'dist_rbf'):
            ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), ctx_edges, atom_mask)
            inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), inter_edges, atom_mask)
            ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
            inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
        else:
            ctx_edge_attr, inter_edge_attr = None, None

        # beta = self.trans_x.var_sched.betas[t][batch_ids] # [N]
        beta = self.trans_x.get_timestamp(t)[batch_ids]  # [N]
        eps_H_pred, eps_X_pred = self.eps_net(
            H_noisy, X_noisy, position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,
            ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr
        )

        if return_states:
            return {
                'H_t': H_noisy,
                'X_t': X_noisy,
                'eps_H_pred': eps_H_pred,
                'eps_X_pred': eps_X_pred,
                'batch_ids': batch_ids
            }

        loss_dict = {}

        # equivariant vector feature loss, TODO: latent channel
        if sample_structure:
            mask_loss = mask_generate[:, None] & atom_mask
            loss_X = F.mse_loss(eps_X_pred[mask_loss], eps_X[mask_loss], reduction='none').sum(dim=-1)  # (Ntgt * n_latent_channel)
            if batch_reduction:
                loss_X = loss_X.sum() / (mask_loss.sum().float() + 1e-8)
            else:
                loss_X = scatter_mean(loss_X, batch_ids[mask_generate], dim=0) / eps_X_pred.shape[-1]
            loss_dict['X'] = loss_X
        else:
            loss_dict['X'] = 0

        # invariant scalar feature loss
        if sample_sequence:
            loss_H = F.mse_loss(eps_H_pred[mask_generate], eps_H[mask_generate], reduction='none').sum(dim=-1)  # [N]
            if batch_reduction:
                loss_H = loss_H.sum() / (mask_generate.sum().float() + 1e-8)
            else:
                loss_H = scatter_mean(loss_H, batch_ids[mask_generate], dim=0) / eps_H_pred.shape[-1]
            loss_dict['H'] = loss_H
        else:
            loss_dict['H'] = 0

        return loss_dict

    @torch.no_grad()
    def sample(self, H, X, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask,
        L=None, sample_structure=True, sample_sequence=True, pbar=False, energy_func=None, energy_lambda=0.01,
        guide_mask=None
    ):
        """
        Args:
            H: contextual hidden states, (N, latent_size)
            X: contextual atomic coordinates, (N, 14, 3)
            L: cholesky decomposition of the covariance matrix \Sigma=LL^T, (bs, 3, 3)
            energy_func: guide diffusion towards lower energy landscape
            guide_mask: fix some part of the generated ligand
        """
        # if L is not None:
        #     L = L / self.std
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        X, centers = self._normalize_position(X, batch_ids, mask_generate, atom_mask, L)
        # print(X[0, 0])

        # save guide part
        if guide_mask is not None:
            batch_size = batch_ids.max() + 1 
            guide_H, guide_X = H.clone().detach(), X.clone().detach()

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            X_rand = torch.randn_like(X) # [N, 14, 3]
            X_init = torch.where(mask_generate[:, None, None].expand_as(X), X_rand, X)
        else:
            X_init = X

        if sample_sequence:
            H_rand = torch.randn_like(H)
            H_init = torch.where(mask_generate[:, None].expand_as(H), H_rand, H)
        else:
            H_init = H

        # traj = {self.num_steps: (self._unnormalize_position(X_init, centers, batch_ids, L), H_init)}
        traj = {self.num_steps: (X_init, H_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            X_t, H_t = traj[t]
            # X_t, _ = self._normalize_position(X_t, batch_ids, mask_generate, atom_mask, L)
            X_t, H_t = torch.round(X_t, decimals=4), torch.round(H_t, decimals=4) # reduce numerical error
            # print(t, 'input', X_t[0, 0] * 1000)
            
            # beta = self.trans_x.var_sched.betas[t].view(1).repeat(X_t.shape[0])
            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            ctx_edges, inter_edges = self._get_edges(mask_generate, batch_ids, lengths)
            if hasattr(self, 'dist_rbf'):
                ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), ctx_edges, atom_mask)
                inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), inter_edges, atom_mask)
                ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
            else:
                ctx_edge_attr, inter_edge_attr = None, None

            eps_H, eps_X = self.eps_net(
                H_t, X_t, position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,
                ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr
            )
            if energy_func is not None:
                with torch.enable_grad():
                    cur_X_state = X_t.clone()
                    cur_X_state.requires_grad = True
                    cur_H_state = H_t.clone()
                    cur_H_state.requires_grad = True
                    energy = energy_func(
                        H=cur_H_state,
                        X=self._unnormalize_position(cur_X_state, centers, batch_ids, L),
                        mask_generate=mask_generate, batch_ids=batch_ids, t=torch.ones_like(lengths) * t)
                    energy_eps_H, energy_eps_X = grad([energy], [cur_H_state, cur_X_state], create_graph=False, retain_graph=False, allow_unused=True)

                if energy_eps_H is not None:
                    energy_eps_H = energy_eps_H.float()
                    energy_eps_H[~mask_generate] = 0
                    energy_eps_H = -energy_eps_H
                if energy_eps_X is not None:
                    energy_eps_X = energy_eps_X.float()
                    energy_eps_X[~mask_generate] = 0
                    energy_eps_X = -energy_eps_X
            else:
                energy_eps_H, energy_eps_X = None, None

            H_next = self.trans_h.denoise(H_t, eps_H, mask_generate, batch_ids, t_tensor, guidance=energy_eps_H, guidance_weight=energy_lambda)
            X_next = self.trans_x.denoise(X_t, eps_X, mask_generate, batch_ids, t_tensor, guidance=energy_eps_X, guidance_weight=energy_lambda)
            
            if guide_mask is not None:
                t_minus_one_batch = torch.full([batch_size, ], fill_value=t - 1, dtype=torch.long, device=X_t.device)
                X_t_minus_1, _ = self.trans_x.add_noise(guide_X, mask_generate, batch_ids, t_minus_one_batch)
                H_t_minus_1, _ = self.trans_h.add_noise(guide_H, mask_generate, batch_ids, t_minus_one_batch)
                H_next = torch.where(guide_mask[:, None].expand_as(H_next), H_t_minus_1, H_next)
                X_next = torch.where(guide_mask[:, None, None].expand_as(X_next), X_t_minus_1, X_next)
                assert not torch.any(torch.isnan(H_next)), f'{t}, H nan'
                assert not torch.any(torch.isnan(X_next)), f'{t}, X nan'

            if not sample_structure:
                X_next = X_t
            if not sample_sequence:
                H_next = H_t

            # traj[t-1] = (self._unnormalize_position(X_next, centers, batch_ids, L), H_next)
            traj[t-1] = (X_next, H_next)
            traj[t] = (self._unnormalize_position(traj[t][0], centers, batch_ids, L).cpu(), traj[t][1].cpu())
            # traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
        traj[0] = (self._unnormalize_position(traj[0][0], centers, batch_ids, L), traj[0][1])
        return traj


    def _vb_terms_bpd(
        self, H_0, X_0, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask, L, t, sample_structure=True, sample_sequence=True
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        states = self(H_0, X_0, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask, L, t, return_states=True)
        true_mean, true_log_variance_clipped, pred_mean, pred_log_variance, start = [], [], [], [], []
        if sample_sequence:
            h_true_mean, _, h_true_log_variance_clipped = self.trans_h.q_posterior_mean_variance(
                p_0=H_0, p_t=states['H_t'], batch_ids=states['batch_ids'], t=t
            )
            h_out = self.trans_h.p_mean_variance(
                p_t=states['H_t'], eps_p=states['eps_H_pred'], batch_ids=states['batch_ids'], t=t
            )
            true_mean.append(h_true_mean)
            true_log_variance_clipped.append(h_true_log_variance_clipped)
            pred_mean.append(h_out['mean'])
            pred_log_variance.append(h_out['log_variance'])
            start.append(H_0)
        if sample_structure:
            x_true_mean, _, x_true_log_variance_clipped = self.trans_x.q_posterior_mean_variance(
                p_0=X_0, p_t=states['X_t'], batch_ids=states['batch_ids'], t=t
            )
            x_out = self.trans_x.p_mean_variance(
                p_t=states['X_t'], eps_p=states['eps_X_pred'], batch_ids=states['batch_ids'], t=t
            )
            true_mean.append(x_true_mean[:, 0]) # latent point
            true_log_variance_clipped.append(x_true_log_variance_clipped[:, 0])
            pred_mean.append(x_out['mean'][:, 0]) # latent point
            pred_log_variance.append(x_out['log_variance'][:, 0])
            start.append(X_0[:, 0])

        if sample_sequence and sample_structure:
            true_mean = torch.cat(true_mean, dim=-1)
            true_log_variance_clipped = torch.cat(true_log_variance_clipped, dim=-1)
            pred_mean = torch.cat(pred_mean, dim=-1)
            pred_log_variance = torch.cat(pred_log_variance, dim=-1)
            start = torch.cat(start, dim=-1)
        else:
            true_mean = true_mean[0]
            true_log_variance_clipped = true_log_variance_clipped[0]
            pred_mean = pred_mean[0]
            pred_log_variance = pred_log_variance[0]
            start = start[0]

        kl = normal_kl(
            true_mean, true_log_variance_clipped, pred_mean, pred_log_variance
        )
        kl = mean_flat(kl, mask_generate, states['batch_ids']) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            start, means=pred_mean, log_scales=0.5 * pred_log_variance
        )
        # assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll, mask_generate, states['batch_ids']) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output

    @torch.no_grad()
    def cal_confidence(self, H_0, X_0, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask, L=None, sample_structure=True, sample_sequence=True):
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        batch_size = batch_ids.max() + 1
        device = H_0.device

        vb = []
        for t in range(2, self.num_steps):
            t_batch = torch.tensor([t] * batch_size, device=device)
            loss_dict = self.forward(
                H_0=H_0,
                X_0=X_0,
                position_embedding=position_embedding,
                mask_generate=mask_generate,
                lengths=lengths,
                atom_embeddings=atom_embeddings,
                atom_mask=atom_mask,
                L=L,
                t=t_batch,
                sample_structure=sample_structure,
                sample_sequence=sample_sequence,
                batch_reduction=False
            )
            t_vb = 0
            if sample_structure:
                struct_vb = loss_dict['X'] * vb_coefficient(self.trans_x, t)
                t_vb += struct_vb
            if sample_sequence:
                seq_vb = loss_dict['H'] * vb_coefficient(self.trans_h, t)
                t_vb += seq_vb
            vb.append(t_vb)
        
        vb = torch.stack(vb, dim=1)
        total_vb = vb.sum(dim=1)
        return total_vb

        vb = []
        for t in list(range(1, self.num_steps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            # Calculate VLB term at the current timestep
            t_vbd = self._vb_terms_bpd(H_0, X_0, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask, L, t_batch, sample_structure, sample_sequence)
            vb.append(t_vbd)
            if t == 0:
                print(t, t_vbd)

        vb = torch.stack(vb, dim=1)
        total_bpd = vb.sum(dim=1)  # discard prior bpd as it is a constant
        print(total_bpd)
        return total_bpd

        # prior_bpd = self._prior_bpd(x_start)
        # total_bpd = vb.sum(dim=1) + prior_bpd
        # return {
        #     "total_bpd": total_bpd,
        #     "prior_bpd": prior_bpd,
        #     "vb": vb,
        # }