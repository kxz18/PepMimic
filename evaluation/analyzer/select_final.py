#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil
from typing import List
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from utils.logger import print_log
from data.converter.blocks_interface import dist_matrix_from_blocks

from evaluation.seq_metric import aar, slide_aar

from .tools import base_parser, _index_content, load_pdb, seq_id, get_pdb_path


ATOM_CONTACT_DIST = 4.0


@dataclass
class Candidate:
    id: str
    seq: str
    src: List[str]
    desc: List[str]
    priority: int=1e10  # -2 for white list, -1 for AF2 plddt > 70, others for min(rank_rosetta, rank_foldx), smaller the better

    def add_src(self, src):
        self.src.append(src)

    def add_desc(self, desc):
        self.desc.append(desc)

    def set_priority(self, prior):
        self.priority = prior

WHITE_PRIO = -2
PLDDT_PRIO = -1

def parse():
    parser = base_parser(desc='Select final candidates')
    parser.add_argument('--white_list', type=str, default=None, help='Specify some necessary candidates')
    parser.add_argument('--elite_index', type=str, default=None, help='Only use references with pdb id in the elite index')
    parser.add_argument('--topk', type=int, default=500, help='Top K for each metric')
    parser.add_argument('--size', type=int, default=384, help='Number of candidates to synthesize')
    parser.add_argument('--ifhit_th', type=float, default=2.0, help='For RFmimicry, it should be 1.0')
    parser.add_argument('--seq_dist_th', type=float, default=0.6, help='Seq id > 0.4 can be clustered')
    return parser.parse_args()


def get_threshold(name, ifhit_th=2.0):
    if name == 'pyrosetta_dG':
        return lambda x: x < -10.0
    elif name == 'foldx_dG':
        return lambda x: x < -5.0
    elif name == 'plddt':
        return lambda x: x >= 70.0
    elif name == 'interface_hit':
        return lambda x: x >= ifhit_th
    else:
        raise NotImplementedError(f'metric {name} not implemented')


def min_better(name):
    return name in ['pyrosetta_dG', 'foldx_dG']


def main(args):

    if args.topk < args.size: args.topk = args.size # topk * number of filters is the pool size

    id2candidates = {}

    # 1. get white list
    if args.white_list is not None:
        with open(args.white_list, 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            _id, seq, desc = line.strip().split('\t')  # id\tseq\tdesc
            id2candidates[_id] = Candidate(_id, seq, ['whitelist'], [desc])
    print_log(f'Load {len(id2candidates)} candidates from white list')


    # 2. get topK with pdb id in the elite index (after threshold)
    ifhit = _index_content(args.root_dir, 'interface_hit')
    ifhit_th_func = get_threshold('interface_hit', ifhit_th=args.ifhit_th)
    if args.elite_index is not None:
        with open(args.elite_index, 'r') as fin:
            lines = fin.readlines()
        qualified_pdbid = {}
        for line in lines:
            pdb_id = line.strip().split('\t')[0]
            qualified_pdbid[pdb_id] = True
    else:
        qualified_pdbid = None
    def pdb_qualify(_id):
        if qualified_pdbid is None:
            return True
        pdb_id = '_'.join(_id.split('_')[:-1])
        return pdb_id in qualified_pdbid

    metrics, ranks = {}, {}
    for name in args.filter_name:
        th_func = get_threshold(name)
        metrics[name] = _index_content(args.root_dir, name)
        qualified_ids = [_id for _id in metrics[name] if ifhit_th_func(ifhit[_id][0]) and th_func(metrics[name][_id][0]) and pdb_qualify(_id)]
        qualified_ids = sorted(qualified_ids, key=lambda _id: metrics[name][_id][0], reverse=not min_better(name))
        if len(qualified_ids) > args.topk:
            qualified_ids = qualified_ids[:args.topk]
        for _id in qualified_ids:
            recommend_desc = f'{name}({round(metrics[name][_id][0], 2)})'
            if _id in id2candidates:
                id2candidates[_id].add_src(name)
                id2candidates[_id].add_desc(recommend_desc)
            else:
                id2candidates[_id] = Candidate(_id, metrics[name][_id][1], [name], [recommend_desc])
        ranks[name] = { _id: i for i, _id in enumerate(qualified_ids) }
        print_log(f'{len(qualified_ids)} qualified for metric {name}')


    # 3. cluster
    seq2id = {}
    for _id in id2candidates:
        seq2id[id2candidates[_id].seq] = _id
    print_log(f'Unique sequences {len(seq2id)}, all candidates {len(id2candidates)} (delta is the repeative sequence)')
    
    seq_dist_th = args.seq_dist_th  # sequences above 40% match rate can be clustered together
    print_log(f'Calculating sequence identities...')
    dists = []
    for i, seq1 in enumerate(tqdm(seq2id)):
        dists.append([])
        for j, seq2 in enumerate(seq2id):
            _, sim = seq_id(seq1, seq2)
            dists[i].append(1 - sim)
    dists = np.array(dists)
    Z = linkage(squareform(dists), 'single')
    cluster = fcluster(Z, t=seq_dist_th, criterion='distance') # [N_seq]

    clu2ids = {}
    for c, seq in zip(cluster, seq2id):
        _id = seq2id[seq]
        if c not in clu2ids: clu2ids[c] = []
        clu2ids[c].append(_id)

    print_log(f'Number of clusters: {len(clu2ids)}')

    # 4. select the representative from each cluster
    selected_candidates = []
    for c in clu2ids:
        ids = clu2ids[c]
        done = False
        # priority 1: in the white list
        if not done:
            for _id in ids:
                cand = id2candidates[_id]
                if 'whitelist' in cand.src:
                    cand.set_priority(WHITE_PRIO)
                    selected_candidates.append(cand)
                    done = True
        # priority 2: af2 plddt > 70
        if not done:
            if 'plddt' not in metrics:
                print_log(f'pLDDT results not found', level='WARN')
                metrics['plddt'] = {}
            has_plddt_ids = [_id for _id in ids if _id in metrics['plddt']]
            if len(has_plddt_ids):
                best_id = max(has_plddt_ids, key=lambda _id: metrics['plddt'][_id][0])
                if metrics['plddt'][best_id][0] > 70.0:
                    cand = id2candidates[best_id]
                    cand.set_priority(PLDDT_PRIO)
                    selected_candidates.append(cand)
                    done = True
        # priority 3: best rank of rosetta or foldx
        if not done:
            id2rank = {}
            for _id in ids:
                id2rank[_id] = min(ranks.get('pyrosetta_dG', {}).get(_id, 1e10), ranks.get('foldx_dG', {}).get(_id, 1e10))
            best_id = min(ids, key=lambda _id: id2rank[_id])
            cand = id2candidates[best_id]
            cand.set_priority(id2rank[_id])
            selected_candidates.append(cand)
            done = True
        assert done
        selected_candidates[-1].add_desc(f'cluster size({len(ids)})')
    print_log(f'Number of qualified candidates after filtering: {len(selected_candidates)}')

    # 5. select final candidates
    selected_candidates = sorted(selected_candidates, key=lambda cand: cand.priority)
    selected_candidates = selected_candidates[:args.size]

    # print results
    print_log(f'Candidates sorted, get top {args.size}')
    for i, cand in enumerate(selected_candidates):
        print(i, cand)
    
    # create output directory
    out_dir = os.path.join(args.root_dir, 'final_output')
    pdb_dir = os.path.join(out_dir, 'pdbs')
    if os.path.exists(pdb_dir):
        shutil.rmtree(pdb_dir)
    os.makedirs(pdb_dir)

    # 6. write done results
    # write txt
    with open(os.path.join(out_dir, 'final_candidates.txt'), 'w') as fout:
        for cand in selected_candidates:
            line = f'{cand.seq}\t{len(cand.seq)}\t{";".join([cand.id] + cand.desc)}\n'
            fout.write(line)

    # 7. write pdbs
    for cand in selected_candidates:
        pdb_path = get_pdb_path(args.root_dir, cand.id, '_openmm')
        os.system(f'cp {pdb_path} {os.path.join(pdb_dir, cand.id + ".pdb")}')


if __name__ == '__main__':
    main(parse())
