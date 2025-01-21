#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import pi, cos

import torch
from torch_scatter import scatter_mean

from .abs_trainer import Trainer
from utils import register as R


@R.register('IFTrainer')
class IFTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict, criterion: str='AAR'):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.max_step = self.config.max_epoch * len(self.train_loader)
        self.criterion = criterion
        assert criterion in ['AAR', 'RMSD', 'Loss'], f'Criterion {criterion} not implemented'
        self.rng_state = None

    ########## Override start ##########

    def _valid_epoch_begin(self, device):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    def share_step(self, batch, batch_idx, val=False):
        results = self.model(**batch)
        if self.is_oom_return(results):
            return results
        loss, (pos_dist, neg_dist) = results

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val, batch_size=pos_dist.shape[0])
        self.log(f'PosDist/{log_type}', pos_dist.mean(), batch_idx, val, batch_size=pos_dist.shape[0])
        self.log(f'NegDist/{log_type}', neg_dist.mean(), batch_idx, val, batch_size=pos_dist.shape[0])

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val=False)

        return loss
    
    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)
    
    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########