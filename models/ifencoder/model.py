#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean

from data.format import VOCAB
from utils import register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import variadic_meshgrid

from ..LDM.ldm import LDMPepDesign
from ..dyMEAN.modules.am_egnn import AMEGNN # adaptive-multichannel egnn
from .ept import EPT

def create_encoder(
    name,
    atom_embed_size,
    embed_size,
    hidden_size,
    edge_size,
    n_channel,
    n_layers,
    dropout,
    n_rbf,
    cutoff
):
    if name == 'dyMEAN':
        encoder = AMEGNN(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=edge_size, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
    elif name == 'EPT':
        encoder = EPT(
            hidden_size, hidden_size, n_rbf, cutoff, edge_size=edge_size,
            n_layers=n_layers
        )
    else:
        raise NotImplementedError(f'Encoder {encoder} not implemented')

    return encoder


def std_conserve_scatter_sum(src, index, dim):
    ones = torch.ones_like(index)
    n = scatter_sum(ones, index, dim=0) # [N]
    value = scatter_sum(src, index, dim=dim) # [N, ...]
    value = value / torch.sqrt(n).unsqueeze(-1)
    return value


@R.register('IFEncoder')
class IFEncoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            out_size,
            ldm_ckpt,
            subgraph_sample_lb=0.5,
            subgraph_sample_ub=1.0,
            n_layers=3,
            dropout=0.1,
            n_rbf=0,
            cutoff=0,
            neg_thresh=10,  # https://github.com/LPDI-EPFL/masif/blob/master/source/masif_modules/MaSIF_ppi_search.py#L140
            encoder='EPT'
        ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.subgraph_sample_lb = subgraph_sample_lb
        self.subgraph_sample_ub = subgraph_sample_ub
        self.neg_thresh = neg_thresh
        
        self.ldm: LDMPepDesign = torch.load(ldm_ckpt, map_location='cpu')
        for param in self.ldm.parameters():
            param.requires_grad = False
        self.ldm.eval()
        embed_size = self.ldm.autoencoder.latent_size
        self.embedding = nn.Linear(embed_size + 3, hidden_size)

        atom_embed_size = hidden_size // 4
        self.atom_embed = nn.Parameter(torch.randn(atom_embed_size, requires_grad=True))

        self.encoder = create_encoder(
            name = encoder,
            atom_embed_size = atom_embed_size,
            embed_size = embed_size,
            hidden_size = hidden_size,
            edge_size = 0,  # two tower, so there is only one type of edge
            n_channel = 1,
            n_layers = n_layers,
            dropout = dropout,
            n_rbf = n_rbf,
            cutoff = cutoff,
        )
        self.linear_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size)
        )

    @torch.no_grad()
    def get_edges(self, batch_ids):    
        lengths = scatter_sum(torch.ones_like(batch_ids), batch_ids, dim=0)
        # edges
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)

        edges = torch.stack([row, col], dim=0) # [2, Ec]

        return edges
    
    @torch.no_grad()
    def prepare_inputs(self, X, S, mask, position_ids, lengths, atom_mask, L=None):
        self.ldm.eval()
        H_0, Z_0, _, _ = self.ldm.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=True) # only residues in ligands remains
        # normalize through linear transformation
        X = X.clone()
        X[mask] = self.ldm.autoencoder._fill_latent_channels(Z_0)
        batch_ids = self.get_batch_ids(S, lengths)
        Z_0, _ = self.ldm.diffusion._normalize_position(
            X, batch_ids, mask, atom_mask, L
        )
        Z_0 = Z_0[mask][:, 0:1]
        batch_ids = batch_ids[mask]
        # sample t
        batch_size = lengths.shape[0]

        return H_0, Z_0, batch_ids, batch_size
    
    @torch.no_grad()
    def get_batch_ids(self, S, lengths):
        batch_ids = torch.zeros_like(S)
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        return batch_ids
    
    @torch.no_grad()
    def get_neg_perm(self, batch_size, device):
        perm = torch.randperm(batch_size, device=device)
        ordered = torch.arange(batch_size, device=device)
        n_tries = 0
        while torch.any(perm == ordered) and n_tries < 10:
            perm = torch.randperm(batch_size, device=device)
            n_tries += 1
        return perm
    
    def encode(self, Z, H, batch_ids, L, t, sample=False, add_noise=False, unnormalize=True, return_sample_mask=False):
        if sample:
            with torch.no_grad():
                ratio = torch.rand_like(batch_ids, dtype=torch.float) * (self.subgraph_sample_ub - self.subgraph_sample_lb) + self.subgraph_sample_lb
                subgraph_sample = torch.rand_like(batch_ids, dtype=torch.float) < ratio[batch_ids]
                while torch.any(scatter_sum(subgraph_sample, batch_ids, dim=0) == 0): 
                    subgraph_sample = torch.rand_like(batch_ids, dtype=torch.float) < ratio[batch_ids]
                Z, H, batch_ids = Z[subgraph_sample], H[subgraph_sample], batch_ids[subgraph_sample]

        # at timestep t
        if add_noise:
            with torch.no_grad():
                mask = torch.ones_like(batch_ids).bool()
                H, _ = self.ldm.diffusion.trans_h.add_noise(H, mask, batch_ids, t)
                Z, _ = self.ldm.diffusion.trans_x.add_noise(Z, mask, batch_ids, t)
            assert unnormalize  # noise is added in the standard space, so unnormalize should be necessary
        beta = self.ldm.diffusion.trans_x.get_timestamp(t)[batch_ids]  # [N]
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        H = torch.cat([H, t_embed], dim=-1)

        # unnormalize (no need to get to the original center)
        if unnormalize:
            centers = torch.zeros_like(Z)
            Z = self.ldm.diffusion._unnormalize_position(Z, centers, batch_ids, L)

        edges = self.get_edges(batch_ids)

        H, pred_X = self.encoder(
                        self.embedding(H), Z, batch_ids, edges,
                    )
        # H = scatter_mean(H, batch_ids, dim=0) # [bs, hidden_size]
        H = std_conserve_scatter_sum(H, batch_ids, dim=0) # [bs, hidden_size]
        H = self.linear_out(H)

        if return_sample_mask:
            return H, subgraph_sample
        return H
    
    @torch.no_grad()
    def get_pos_neg_mask(self, mask, batch_ids):
        pos_mask = torch.zeros_like(mask)
        neg_mask = torch.zeros_like(mask)
        while torch.any(scatter_sum(pos_mask, batch_ids, dim=0) == 0) or torch.any(scatter_sum(neg_mask, batch_ids, dim=0) == 0):
            rand_number = torch.rand_like(mask, dtype=torch.float)  # [N]
            pos_mask = rand_number < 0.5
            neg_mask = ~pos_mask
        return pos_mask, neg_mask

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, L=None, t=None, return_overlap=False):
        H_0, Z_0, batch_ids, batch_size = self.prepare_inputs(
            X, S, mask, position_ids, lengths, atom_mask, L
        )
        if t is None:
            t = torch.randint(0, self.ldm.diffusion.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)

        pos_mask, neg_mask = self.get_pos_neg_mask(mask[mask], batch_ids)
        if return_overlap:
            ref_H, ref_sample_mask = self.encode(Z_0[pos_mask], H_0[pos_mask], batch_ids[pos_mask], L, t, sample=True, add_noise=True, return_sample_mask=True) # [batch_size, hidden]
            pos_H, pos_sample_mask = self.encode(Z_0[pos_mask], H_0[pos_mask], batch_ids[pos_mask], L, t, sample=True, add_noise=True, return_sample_mask=True) # [batch_size, hidden]
            overlap = scatter_sum((ref_sample_mask == pos_sample_mask).long(), batch_ids[pos_mask], dim=0)
        else:
            ref_H = self.encode(Z_0[pos_mask], H_0[pos_mask], batch_ids[pos_mask], L, t, sample=True, add_noise=True) # [batch_size, hidden]
            pos_H = self.encode(Z_0[pos_mask], H_0[pos_mask], batch_ids[pos_mask], L, t, sample=True, add_noise=True) # [batch_size, hidden]
        neg_H = self.encode(Z_0[neg_mask], H_0[neg_mask], batch_ids[neg_mask], L, t, sample=True, add_noise=True) # [batch_size, hidden]
        # ref_H = self.encode(Z_0, H_0, batch_ids, L, t, sample=True, add_noise=True) # [batch_size, hidden]
        # pos_H = self.encode(Z_0, H_0, batch_ids, L, t, sample=True, add_noise=True)  # [batch_size, hidden]
        # neg_perm = self.get_neg_perm(batch_size, pos_H.device)
        # neg_H = self.encode(Z_0, H_0, batch_ids, L, t[neg_perm], sample=True, add_noise=True)[neg_perm] # [batch_size, hidden]

        pos_dist, neg_dist = torch.norm(ref_H - pos_H, dim=-1), torch.norm(ref_H - neg_H, dim=-1)
        # pos_dist = F.cosine_similarity(ref_H, pos_H, dim=-1)
        # neg_dist = F.cosine_similarity(ref_H, neg_H, dim=-1)

        # d-prime loss
        # loss = torch.std(pos_dist, dim=-1) + torch.std(neg_dist, dim=-1) + torch.mean(pos_dist, dim=-1) + torch.mean(F.relu(self.neg_thresh - neg_dist), dim=-1)
        loss = torch.mean(pos_dist, dim=-1) + torch.mean(F.relu(self.neg_thresh - neg_dist), dim=-1)

        if return_overlap:
            return loss, (pos_dist, neg_dist), overlap
        return loss, (pos_dist, neg_dist) 

    def interface_dist(self, t, H, X, mask, batch_ids, ref_H, ref_X, ref_batch_ids, ref_L, **kwargs):
        ori_ref_H = ref_H.clone()
        ref_H = self.encode(ref_X, ref_H, ref_batch_ids, ref_L, t, sample=False, add_noise=True) # [batch_size, hidden]
        gen_H = self.encode(X[mask][:, 0], H[mask], batch_ids[mask], None, t, sample=False, add_noise=False, unnormalize=False)
        self.cache_dist = torch.norm(ref_H - gen_H, dim=-1) 
        if hasattr(self, 'cache_dist_all'):
            # interface similarity if directly use the current state as output
            t = torch.zeros_like(t)
            ref_H = self.encode(ref_X, ori_ref_H, ref_batch_ids, ref_L, t, sample=False, add_noise=True) # [batch_size, hidden]
            gen_H = self.encode(X[mask][:, 0], H[mask], batch_ids[mask], None, t, sample=False, add_noise=False, unnormalize=False)
            self.cache_dist_all.append(torch.norm(ref_H - gen_H, dim=-1))
        return self.cache_dist.sum()

    @torch.no_grad()
    def sample(
        self,
        X, S, mask, position_ids, lengths, atom_mask,
        ref_X, ref_S, ref_mask, ref_position_ids, ref_lengths, ref_atom_mask,
        L=None, ref_L=None,
        sample_opt={
            'pbar': False,
            'energy_func': None,
            'energy_lambda': 0.0
        },
        return_tensor=False,
        optimize_sidechain=True,
        idealize=False,
    ):
        ref_H_0, ref_Z_0, ref_batch_ids, ref_batch_size = self.prepare_inputs(
            ref_X, ref_S, ref_mask, ref_position_ids, ref_lengths, ref_atom_mask, ref_L
        )
        # replace external energy function
        return_all_if_dist = sample_opt.pop('return_all_if_dist', False)
        if return_all_if_dist:
            self.cache_dist_all = []
        if sample_opt.pop('disable_interface_guidance', False):
            self.cache_dist = torch.zeros(ref_batch_size)
        else:
            self.ldm.external_energy = partial(self.interface_dist, ref_X=ref_Z_0, ref_H=ref_H_0, ref_batch_ids=ref_batch_ids, ref_L=ref_L)
        sample_X, sample_S, sample_ppl = self.ldm.sample(X, S, mask, position_ids, lengths, atom_mask, L, sample_opt, return_tensor, optimize_sidechain, idealize)
        if return_all_if_dist:
            if_dist = torch.stack(self.cache_dist_all, dim=1) # [bs, T]
        else:
            if_dist = self.cache_dist
        return sample_X, sample_S, if_dist.tolist()