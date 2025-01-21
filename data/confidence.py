
import os
from typing import Optional, Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from utils import register as R
from utils.const import sidechain_atoms

from data.converter.list_blocks_to_pdb import list_blocks_to_pdb

from .format import VOCAB, Block, Atom
from .mmap_dataset import MMAPDataset
from .converter.blocks_to_data import blocks_to_data_simple



def calculate_covariance_matrix(point_cloud):
    # Calculate the covariance matrix of the point cloud
    covariance_matrix = np.cov(point_cloud, rowvar=False)
    return covariance_matrix


@R.register('ConfidenceDataset')
class ConfidenceDataset(MMAPDataset):

    MAX_N_ATOM = 14

    def __init__(
            self,
            mmap_dir: str,
            backbone_only: bool = False,  # only backbone (N, CA, C, O) or full-atom
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            noise_scale: float = 0.0
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.backbone_only = backbone_only
        self.filtering()
        self._lengths = [len(prop[-2].split(',')) + int(prop[1]) for prop in self._properties]
        self.noise_scale = noise_scale

    def filtering(self):
        pocket_len = [len(prop[-2].split(',')) for prop in self._properties]
        self.used_idx = [i for i in range(len(pocket_len)) if pocket_len[i] > 0]

    def get_len(self, idx):
        idx = self.used_idx[idx]
        return self._lengths[idx]

    def __len__(self):
        return len(self.used_idx)


    def get_summary(self, idx: int):
        idx = self.used_idx[idx]
        props = self._properties[idx]
        _id = self._indexes[idx][0].split('.')[0]
        ref_pdb = os.path.join(self.mmap_dir, 'pdbs', _id + '.pdb')
        rec_chain, lig_chain = props[4], props[5]
        return _id, ref_pdb, rec_chain, lig_chain

    def __getitem__(self, idx: int):
        idx = self.used_idx[idx]
        rec_blocks, lig_blocks = super().__getitem__(idx)
        try:
            pocket_idx = [int(i) for i in self._properties[idx][-2].split(',')]
        except:
            pocket_idx = []
        rec_blocks = [rec_blocks[i] for i in pocket_idx]
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks]

    

        rmsd = float(self._properties[idx][-1])
        _id = self._indexes[idx][0].split('.')[0]
        data_dir = os.path.abspath(os.path.join(self.mmap_dir, '..'))
        pdb_id = _id.split('_')[0]
        if 'gen' in _id:
            mid_dir = os.path.join('candidates', pdb_id)
        else:
            mid_dir = 'references'
        name = os.path.join(data_dir, mid_dir, _id + '.pdb')
        item = blocks_to_data_simple(rec_blocks, lig_blocks)
        item['label'] = [rmsd]
        item['name'] = name
        item['pdb_id'] = pdb_id
        if self.noise_scale > 1e-2:
            block_lengths = np.array(item['block_lengths'])
            segment_ids = np.array(item['segment_ids'])
            rec_len = block_lengths[segment_ids == 0].sum()
            X = np.array(item['X'])
            X_rec, X_lig = X[:rec_len], X[rec_len:]
            noise = np.random.randn(*X_lig.shape) * self.noise_scale
            X_n = np.concatenate([X_rec, X_lig + noise], axis = 0).tolist()
            item['X'] = X_n
        
        return item

    @classmethod
    def collate_fn(cls, batch):
        results = {
            'X': torch.cat([torch.tensor(item['X'], dtype=torch.float) for item in batch], dim=0),
            'B': torch.cat([torch.tensor(item['B'], dtype=torch.long) for item in batch], dim=0),
            'A': torch.cat([torch.tensor(item['A'], dtype=torch.long) for item in batch], dim=0),
            'atom_positions': torch.cat([torch.tensor(item['atom_positions'], dtype=torch.long) for item in batch], dim=0),
            'block_lengths': torch.cat([torch.tensor(item['block_lengths'], dtype=torch.long) for item in batch], dim=0),
            'segment_ids': torch.cat([torch.tensor(item['segment_ids'], dtype=torch.long) for item in batch], dim=0),
            'lengths': torch.tensor([len(item['B']) for item in batch], dtype=torch.long),
            'label': torch.cat([torch.tensor(item['label'], dtype=torch.float) for item in batch], dim=0),
            'name': [item['name'] for item in batch],
            'pdb_id': [item['pdb_id'] for item in batch],
        }

        results['X'] = results['X'].unsqueeze(-2)  # number of channel is 1
        return results

@R.register('BalancedConfidenceDataset')
class BalancedConfidenceDataset(torch.utils.data.Dataset):

    MAX_N_ATOM = 14

    def __init__(
            self,
            gen_mmap_dir: str,
            ref_mmap_dir: str,
            pos_noise_scale: float = 1.0,
            neg_noise_scale: float = 5.0,
            threshold: float = 3.0
        ) -> None:
        self.gen_dataset = MMAPDataset(gen_mmap_dir)
        self.ref_dataset = MMAPDataset(ref_mmap_dir)
        self.pos_noise_scale = pos_noise_scale
        self.neg_noise_scale = neg_noise_scale
        self.threshold = threshold
        self.preparing()

    def preparing(self):
        self.ref_dict = {}
        gen_pocket_len = [len(prop[-2].split(',')) for prop in self.gen_dataset._properties]
        ref_pocket_len = [len(prop[-2].split(',')) for prop in self.ref_dataset._properties]
        gen_ligand_len = [int(prop[1]) for prop in self.gen_dataset._properties]
        ref_ligand_len = [int(prop[1]) for prop in self.ref_dataset._properties]
        gen_rmsds = [float(prop[-1]) for prop in self.gen_dataset._properties]
        for ref_idx, ref_name in enumerate(self.ref_dataset._indexes):
            _id = '_'.join(ref_name[0].split('.')[0].split('_')[:-1])
            if ref_pocket_len[ref_idx] > 0:
                self.ref_dict[_id] = ref_idx
        self.pdb_ids = list(self.ref_dict.keys())
        self.gen_dict = {k:[] for k in self.pdb_ids}
        for gen_idx, gen_name in enumerate(self.gen_dataset._indexes):
            _id = '_'.join(gen_name[0].split('.')[0].split('_')[:-2])
            if _id in self.gen_dict and gen_pocket_len[gen_idx] > 0:
                self.gen_dict[_id].append(gen_idx)

        tot_len = len(self.pdb_ids) * 22

        self.cases = [(0, 0, 0) for _ in range(tot_len)]
        """
        cases: case, idx, rmsd
        0 - ground truth
        1 - gen
        2 - neg from ground truth
        3 - pos from ground truth
        """
        self._lengths = [0 for _ in range(tot_len)]

        def set_cases(idx, case, ori_idx, rmsd, length):
            self.cases[idx] = (case, ori_idx, rmsd)
            self._lengths[idx] = length

        for sample_idx, pid in enumerate(self.pdb_ids):

            ref_idx = self.ref_dict[pid]
            ref_length = ref_pocket_len[ref_idx] + ref_ligand_len[ref_idx]

            set_cases(sample_idx * 22, 0, ref_idx, 0.0, ref_length)
            set_cases(sample_idx * 22 + 1, 2, ref_idx, self.neg_noise_scale, ref_length)

            cur_gen_idx = 0

            for gen_idx in self.gen_dict[pid]:
                # print(cur_gen_idx)
                gen_length = gen_pocket_len[gen_idx] + gen_ligand_len[gen_idx]
                gen_rmsd = gen_rmsds[gen_idx]
                set_cases(sample_idx * 22 + 2 + cur_gen_idx, 1, gen_idx, gen_rmsd, gen_length)
                aug_type = 2 if gen_rmsd <= self.threshold else 3
                aug_rmsd = self.neg_noise_scale if gen_rmsd <= self.threshold else self.pos_noise_scale
                set_cases(sample_idx * 22 + 12 + cur_gen_idx, aug_type, ref_idx, aug_rmsd, ref_length)
                cur_gen_idx += 1

            while cur_gen_idx < 10:

                set_cases(sample_idx * 22 + 2 + cur_gen_idx, 2, ref_idx, self.neg_noise_scale, ref_length)
                set_cases(sample_idx * 22 + 12 + cur_gen_idx, 3, ref_idx, self.pos_noise_scale, ref_length)
                cur_gen_idx += 1




    def get_len(self, idx):
        return self._lengths[idx]

    def __len__(self):
        return len(self._lengths)


    def get_summary(self, idx: int):
        cs, ori_idx, rmsd = self.cases[idx]
        dataset = self.gen_dataset if cs == 1 else self.ref_dataset

        props = dataset._properties[ori_idx]
        _id = dataset._indexes[ori_idx][0].split('.')[0]
        ref_pdb = os.path.join(dataset.mmap_dir, 'pdbs', _id + '.pdb')
        rec_chain, lig_chain = props[4], props[5]
        return _id, ref_pdb, rec_chain, lig_chain

    def __getitem__(self, idx: int):

        cs, ori_idx, rmsd = self.cases[idx]
        dataset = self.gen_dataset if cs == 1 else self.ref_dataset
        rec_blocks, lig_blocks = dataset.__getitem__(ori_idx)
        try:
            pocket_idx = [int(i) for i in dataset._properties[ori_idx][-2].split(',')]
        except:
            pocket_idx = []
        rec_blocks = [rec_blocks[i] for i in pocket_idx]
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks]

        item = blocks_to_data_simple(rec_blocks, lig_blocks)
        item['label'] = [rmsd]

        if cs == 2: # neg from gt
            block_lengths = np.array(item['block_lengths'])
            segment_ids = np.array(item['segment_ids'])
            rec_len = block_lengths[segment_ids == 0].sum()
            X = np.array(item['X'])
            X_rec, X_lig = X[:rec_len], X[rec_len:]
            noise = np.random.randn(3) * self.neg_noise_scale
            X_n = np.concatenate([X_rec, X_lig + noise], axis = 0).tolist()
            item['X'] = X_n            
        if cs == 3: # pos from gt
            block_lengths = np.array(item['block_lengths'])
            segment_ids = np.array(item['segment_ids'])
            rec_len = block_lengths[segment_ids == 0].sum()
            X = np.array(item['X'])
            X_rec, X_lig = X[:rec_len], X[rec_len:]
            noise = np.random.randn(*X_lig.shape) * self.pos_noise_scale
            X_n = np.concatenate([X_rec, X_lig + noise], axis = 0).tolist()
            item['X'] = X_n
        
        return item

    @classmethod
    def collate_fn(cls, batch):
        results = {
            'X': torch.cat([torch.tensor(item['X'], dtype=torch.float) for item in batch], dim=0),
            'B': torch.cat([torch.tensor(item['B'], dtype=torch.long) for item in batch], dim=0),
            'A': torch.cat([torch.tensor(item['A'], dtype=torch.long) for item in batch], dim=0),
            'atom_positions': torch.cat([torch.tensor(item['atom_positions'], dtype=torch.long) for item in batch], dim=0),
            'block_lengths': torch.cat([torch.tensor(item['block_lengths'], dtype=torch.long) for item in batch], dim=0),
            'segment_ids': torch.cat([torch.tensor(item['segment_ids'], dtype=torch.long) for item in batch], dim=0),
            'lengths': torch.tensor([len(item['B']) for item in batch], dtype=torch.long),
            'label': torch.cat([torch.tensor(item['label'], dtype=torch.float) for item in batch], dim=0),
        }

        results['X'] = results['X'].unsqueeze(-2)  # number of channel is 1
        return results




if __name__ == '__main__':
    import sys
    dataset = ConfidenceDataset(sys.argv[1], backbone_only=True)
    print(dataset[0])