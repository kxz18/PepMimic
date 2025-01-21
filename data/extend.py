
import os
import requests
from typing import List, Optional, Dict
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import torch

from utils import register as R
from utils.const import sidechain_atoms
from utils.logger import print_log

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_cb_interface, blocks_interface
from utils.file_utils import get_filename

from .codesign import calculate_covariance_matrix
from .format import VOCAB, Block, Atom

from data.converter.blocks_to_data import blocks_to_data_simple
from data.mimicry import Complex, scan


@R.register('ExtendDataset')
class ExtendDataset(torch.utils.data.Dataset):
    
    MAX_N_ATOM = 14
    
    def __init__(self, ref_dir, n_sample_per_cplx, length_lb, length_ub):
        '''
        Args:
            ref_dir: str, directory containing index.txt and pdb files
            n_sample_per_cplx: int, number of mimicry samples for each complex
            length_lb: int, number of residues >= length_lb
            length_ub: int, number of residues <= length_ub
        '''
        super().__init__()
        self.complexes = scan(ref_dir, full_ligand=True) # the ligand is a peptide that needs to be extended
        print_log(f'Successfully scanned {len(self.complexes)} reference complexes')
        self.n_sample_per_cplx = n_sample_per_cplx
        self.length_lb = length_lb
        self.length_ub = length_ub
        self.sample_lengths = np.random.randint(length_lb, length_ub + 1, size=len(self))
        self.cand_save_dir = os.path.join(ref_dir, 'results')

    def get_summary(self, idx):
        cplx = self.complexes[idx % len(self.complexes)]
        return cplx, self.sample_lengths[idx] + len(cplx.lig_blocks)

    def __len__(self):
        return len(self.complexes) * self.n_sample_per_cplx

    @classmethod
    def _form_data(cls, rec_blocks, lig_blocks):
        mask = [0 for _ in rec_blocks] + [1 for _ in lig_blocks]
        position_ids = [block.id[0] for block in rec_blocks + lig_blocks]
        X, S, atom_mask = [], [], []
        for block in rec_blocks + lig_blocks:
            symbol = VOCAB.abrv_to_symbol(block.abrv)
            atom2coord = { unit.name: unit.get_coord() for unit in block.units }
            bb_pos = np.mean(list(atom2coord.values()), axis=0).tolist()
            coords, coord_mask = [], []
            for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(symbol, []):
                if atom_name in atom2coord:
                    coords.append(atom2coord[atom_name])
                    coord_mask.append(1)
                else:
                    coords.append(bb_pos)
                    coord_mask.append(0)
            n_pad = cls.MAX_N_ATOM - len(coords)
            for _ in range(n_pad):
                coords.append(bb_pos)
                coord_mask.append(0)

            X.append(coords)
            S.append(VOCAB.symbol_to_idx(symbol))
            atom_mask.append(coord_mask)
        
        X, atom_mask = torch.tensor(X, dtype=torch.float), torch.tensor(atom_mask, dtype=torch.bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        cov = calculate_covariance_matrix(X[~mask][:, 1][atom_mask[~mask][:, 1]].numpy())
        eps = 1e-4
        cov = cov + eps * np.identity(cov.shape[0])
        L = torch.from_numpy(np.linalg.cholesky(cov)).float().unsqueeze(0)

        item =  {
            'X': X,                                                         # [N, 14] or [N, 4] if backbone_only == True
            'S': torch.tensor(S, dtype=torch.long),                         # [N]
            'position_ids': torch.tensor(position_ids, dtype=torch.long),   # [N]
            'mask': mask,                                                   # [N], 1 for generation
            'atom_mask': atom_mask,                                         # [N, 14] or [N, 4], 1 for having records in the PDB
            'lengths': len(S),
            'L': L
        }

        return item

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('Out of range')
        cplx = self.complexes[idx % len(self.complexes)]
        cplx.check_interface()

        gen_len = self.sample_lengths[idx]
        pep_placeholder = [Block(
            'GLY', [Atom('CA', [0, 0, 0], 'C')], (i + 1, ' ')
        ) for i in range(gen_len)]
        if cplx.desc == 'N-terminal':
            pep_blocks = pep_placeholder + cplx.lig_blocks
        elif cplx.desc == 'C-terminal':
            pep_blocks = cplx.lig_blocks + pep_placeholder
        else:
            raise NotImplementedError(f'')
        item = self._form_data(cplx.rec_blocks, pep_blocks)
        guide_mask = [0 for _ in cplx.rec_blocks] + [1 for _ in cplx.lig_blocks] + [0 for _ in pep_placeholder]

        item['guide_mask'] = torch.tensor(guide_mask, dtype=torch.bool)
        
        return item

    @classmethod
    def collate_fn(cls, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if values[0] is None:
                results[key] = None
                continue
            if 'lengths' in key:
                results[key] = torch.tensor(values, dtype=torch.long)
            else:
                results[key] = torch.cat(values, dim=0)
        return results
    

if __name__ == '__main__':
    import sys
    dataset = ExtendDataset(sys.argv[1], 5, 5, 6)
    print(len(dataset))

    for i, item in enumerate(dataset):
        print(i, item)
        break