
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



@dataclass
class Complex:
    pdb_id: str
    file_path: str
    rec_chains: List[str]
    lig_chains: List[str]
    desc: Optional[str] = None
    rec_blocks: Optional[List[Block]] = None
    lig_blocks: Optional[List[Block]] = None
    cache_data: Optional[Dict] = None

    def check_interface(self, full_ligand=False):
        if self.rec_blocks is not None and self.lig_blocks is not None:
            return
        # calculate interface
        name2chains = pdb_to_list_blocks(
            self.file_path, selected_chains=self.rec_chains + self.lig_chains, dict_form=True
        )
        rec_blocks, lig_blocks = [], []
        for chain in self.rec_chains:
            rec_blocks.extend(name2chains[chain])
        for chain in self.lig_chains:
            lig_blocks.extend(name2chains[chain])
        _, (pocket_idx, _) = blocks_cb_interface(rec_blocks, lig_blocks, 10.0)  # pocket threshold, CB 10A
        rec_blocks = [rec_blocks[i] for i in pocket_idx]
        for block, i in zip(rec_blocks, pocket_idx):
            block.id = (i + 1, ' ')
        _, (_, lig_idx) = blocks_interface(rec_blocks, lig_blocks, 6.0) # full-atom dist 6.0A, precise contact on ligand
        if not full_ligand:
            lig_blocks = [lig_blocks[i] for i in lig_idx]

        self.rec_blocks, self.lig_blocks = rec_blocks, lig_blocks

        return len(self.rec_blocks) > 0 and len(self.lig_blocks) > 0

    def get_cache_data(self):
        return self.cache_data
    
    def set_cache_data(self, data):
        self.cache_data = data


def download(pdb_id, out_path):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    res = requests.get(url)
    if res.status_code != 200:
        print_log(f'Failed to fetch {pdb_id}', level='WARN')
        return False
    else:
        with open(out_path, 'w') as fout:
            fout.write(res.text)
        return True


def scan(ref_dir, full_ligand=False) -> List[Complex]:
    with open(os.path.join(ref_dir, 'index.txt'), 'r') as fin:
        lines = fin.readlines()
    summaries = []
    for line in tqdm(lines):
        pdb, rec_chains, lig_chains, desc = line.strip().split('\t')
        rec_chains, lig_chains = rec_chains.split(','), lig_chains.split(',')
        pdb_file = os.path.join(ref_dir, f'{pdb}.pdb')
        if not os.path.exists(pdb_file):
            ok = download(pdb, pdb_file)
            if not ok:
                continue
        cplx = Complex(
            pdb_id=pdb, file_path=pdb_file,
            rec_chains=rec_chains,
            lig_chains=lig_chains,
            desc=desc
        )
        if cplx.check_interface(full_ligand):
            summaries.append(cplx)
    return summaries


@R.register('MimicryDataset')
class MimicryDataset(torch.utils.data.Dataset):
    
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
        self.complexes = scan(ref_dir)
        print_log(f'Successfully scanned {len(self.complexes)} reference complexes')
        self.n_sample_per_cplx = n_sample_per_cplx
        self.length_lb = length_lb
        self.length_ub = length_ub
        self.sample_lengths = np.random.randint(length_lb, length_ub + 1, size=len(self))
        self.cand_save_dir = os.path.join(ref_dir, 'results')

    def get_summary(self, idx):
        return self.complexes[idx % len(self.complexes)], self.sample_lengths[idx]

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

        ref_item = cplx.get_cache_data()
        if ref_item is None:
            ref_item = self._form_data(cplx.rec_blocks, cplx.lig_blocks)
            cplx.set_cache_data(ref_item)
        gen_len = self.sample_lengths[idx]
        pep_placeholder = [Block(
            'CYS', [Atom('CA', [0, 0, 0], 'C')], (i + 1, ' ')
        ) for i in range(gen_len)]
        item = self._form_data(cplx.rec_blocks, pep_placeholder)

        for key in ref_item:
            item['ref_' + key] = ref_item[key]

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
    

@R.register('PDBDataset')
class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, summary_list):
        super().__init__()
        '''
        Args:
            summary_list: list of (file_path, receptor_chains, ligand_chains)
        '''
        self.complexes = [
            Complex(get_filename(file_path), file_path, rec_chains, lig_chains)
            for file_path, rec_chains, lig_chains in summary_list
        ]

    def __getitem__(self, idx):
        cplx = self.complexes[idx]
        cplx.check_interface(full_ligand=True)

        item = cplx.get_cache_data()
        if item is None:
            for i, block in enumerate(cplx.lig_blocks):
                block.id = (i + 1, ' ')
            item = MimicryDataset._form_data(cplx.rec_blocks, cplx.lig_blocks)
            cplx.set_cache_data(item)
        
        return item

    def __len__(self):
        return len(self.complexes)

    @classmethod
    def collate_fn(self, batch):
        return MimicryDataset.collate_fn(batch)

@R.register('PDBConfDataset')
class PDBConfDataset(torch.utils.data.Dataset):
    def __init__(self, summary_list):
        super().__init__()
        '''
        Args:
            summary_list: list of (file_path, receptor_chains, ligand_chains)
        '''
        self.complexes = [
            Complex(get_filename(file_path), file_path, rec_chains, lig_chains)
            for file_path, rec_chains, lig_chains in summary_list
        ]

    def __getitem__(self, idx):
        cplx = self.complexes[idx]
        cplx.check_interface(full_ligand=True)

        item = cplx.get_cache_data()
        if item is None:
            for i, block in enumerate(cplx.lig_blocks):
                block.id = (i + 1, ' ')
            # item = MimicryDataset._form_data(cplx.rec_blocks, cplx.lig_blocks)
            item = blocks_to_data_simple(cplx.rec_blocks, cplx.lig_blocks)
            cplx.set_cache_data(item)
        
        return item

    def __len__(self):
        return len(self.complexes)

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
            'label': torch.cat([torch.tensor([0.0], dtype=torch.float) for item in batch], dim=0), # fake label
        }

        results['X'] = results['X'].unsqueeze(-2)  # number of channel is 1
        return results


if __name__ == '__main__':
    import sys
    dataset = MimicryDataset(sys.argv[1], 5, 10, 12)
    print(len(dataset))

    for i, item in enumerate(dataset):
        print(i, item)
        break