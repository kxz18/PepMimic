#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import argparse
from math import sqrt

import numpy as np
from Bio.Align import PairwiseAligner
from munkres import Munkres

from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from evaluation.seq_metric import align_sequences
from evaluation.rmsd import kabsch


def base_parser(desc=''):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory, e.g. ./targets/PD-L1')
    parser.add_argument('--filter_name', type=str, default=['pyrosetta_dG'], nargs='+', choices=['pyrosetta_dG', 'foldx_dG', 'interface_hit', 'plddt'],
                        help='Name of the filter name')
    return parser


def _index_content(root_dir, name, result_dir='results', has_header=False, transform=lambda val: float(val)):
    '''
    Extract results from given file name
    '''
    if not name.endswith('.txt'):
        name = name + '.txt'
    file_path = os.path.join(root_dir, result_dir, name)
    return _index_content_from_file(file_path, has_header, transform)


def _index_content_from_file(file_path, has_header=False, transform=lambda val: float(val)):
    with open(file_path, 'r') as fin:
        lines = fin.readlines()
    if has_header:
        lines = lines[1:]
    contents = {}
    for line in lines:
        line = line.strip().split('\t')
        item_id, value, seq = line[0], line[1], line[-1]
        contents[item_id] = (transform(value), seq)
    return contents


def load_references(root_dir, keep_desc=False):
    with open(os.path.join(root_dir, 'index.txt'), 'r') as fin:
        lines = fin.readlines()
    references = {}
    for line in lines:
        line = line.strip('\n').split('\t')
        if keep_desc:
            references[line[0]] = (line[1].split(','), line[2].split(','), line[3].split(',')) # PDB id, receptor chains, ligand chains, desc
        else:
            references[line[0]] = (line[1].split(','), line[2].split(',')) # PDB id, receptor chains, ligand chains
    return references


def load_pmetric(root_dir, result_dir='results'):
    with open(os.path.join(root_dir, result_dir, 'results.jsonl'), 'r') as fin:
        lines = fin.readlines()
    metrics = {}
    for line in lines:
        item = json.loads(line)
        _id = f'{item["id"]}_{item["number"]}'
        metrics[_id] = (item['pmetric'], item['gen_seq'])
    return metrics


def get_pdb_id(_id):
    return '_'.join(_id.split('_')[:-1])


def get_number(_id):
    return _id.split('_')[-1]


def get_ref_rec_lig_chains(root_dir, pdb_id):
    index_file = os.path.join(root_dir, 'index.txt')
    with open(index_file, 'r') as fin:
        lines = fin.readlines()
    
    for line in lines:
        line = line.strip('\n').split('\t')
        if line[0] == pdb_id:
            return line[1].split(','), line[2].split(',') # rec_chains, lig_chains


def get_rec_lig_chain(root_dir, _id):
    pdb_id = get_pdb_id(_id)
    result_jsonl = os.path.join(root_dir, 'results', 'results.jsonl')
    with open(result_jsonl, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            item = json.loads(line)
            if item['id'] == pdb_id:
                return item['rec_chains'], item['lig_chain']


def get_pdb_path(root_dir, _id, suffix=''):
    pdb_id, number = get_pdb_id(_id), get_number(_id)
    pdb_path = os.path.join(root_dir, 'results', pdb_id, f'{pdb_id}_gen_{number}{suffix}.pdb')
    return pdb_path


def load_pdb(root_dir, _id, suffix='', **kwargs):
    pdb_path = get_pdb_path(root_dir, _id, suffix)
    rec_chains, lig_chain = get_rec_lig_chain(root_dir, _id)

    chain2blocks = pdb_to_list_blocks(pdb_path, selected_chains=rec_chains + [lig_chain], dict_form=True)

    rec_blocks = []
    for c in rec_chains:
        rec_blocks += chain2blocks[c]
    lig_blocks = chain2blocks[lig_chain]
    return rec_blocks, lig_blocks


hydrophobic_residues=['V','I','L','M','F','W','C']
charged_residues = ['K', 'R', 'D', 'E']
polar_residues = ['S', 'T', 'Y',   'N', 'Q', 'H']


def _count_ratio(seq, aa_list):
    in_cnt = {}
    for aa in aa_list:
        in_cnt[aa] = True
    cnt = 0
    for aa in seq:
        if aa in in_cnt:
            cnt += 1
    return cnt / len(seq)


def hydrophobic_ratio(seq):
    return _count_ratio(seq, hydrophobic_residues)


def polar_ratio(seq):
    return _count_ratio(seq, polar_residues)


def charged_ratio(seq):
    return _count_ratio(seq, charged_residues)


def seq_id(sequence_A, sequence_B):
    aligner = PairwiseAligner()
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -1
    alns = aligner.align(sequence_A, sequence_B)

    best_aln = alns[0]
    aligned_A, aligned_B = best_aln

    base = sqrt(aligner.score(sequence_A, sequence_A) * aligner.score(sequence_B, sequence_B))
    seq_id = aligner.score(sequence_A, sequence_B) / base

    return (aligned_A, aligned_B), seq_id


def align_rmsd(pdb1, pdb2, align_map, rmsd_map):
    chain2blocks1 = pdb_to_list_blocks(pdb1, list(align_map.keys()) + list(rmsd_map.keys()), dict_form=True)
    chain2blocks2 = pdb_to_list_blocks(pdb2, list(align_map.values()) + list(rmsd_map.values()), dict_form=True)

    align_X1, align_X2 = [], []
    align_seq1, align_seq2 = '', ''

    for c1 in align_map:
        c2 = align_map[c1]
        for block in chain2blocks1[c1]:
            align_X1.append(block.get_unit_by_name('CA').coordinate)
            align_seq1 += VOCAB.abrv_to_symbol(block.abrv)
        for block in chain2blocks2[c2]:
            align_X2.append(block.get_unit_by_name('CA').coordinate)
            align_seq2 += VOCAB.abrv_to_symbol(block.abrv)
    align_seq1, align_seq2 = align_seq1.replace('?', 'X'), align_seq2.replace('?', 'X')
    (aligned1, aligned2), _ = align_sequences(align_seq1, align_seq2)
    i, j = 0, 0
    hit_X1, hit_X2 = [], []
    assert len(aligned1) == len(aligned2)
    for aa1, aa2 in zip(aligned1, aligned2):
        if aa1 != '-' and aa2 != '-':
            hit_X1.append(align_X1[i])
            hit_X2.append(align_X2[j])
        if aa1 != '-':
            i += 1
        if aa2 != '-':
            j += 1
    hit_X1, hit_X2 = np.array(hit_X1), np.array(hit_X2)
    _, Q, t = kabsch(hit_X1, hit_X2) # turn X1 into X2

    # structure to calculate rmsd
    rmsd_X1, rmsd_X2 = [], []
    for c1 in rmsd_map:
        c2 = rmsd_map[c1]
        for block in chain2blocks1[c1]:
            rmsd_X1.append(block.get_unit_by_name('CA').coordinate)
        for block in chain2blocks2[c2]:
            rmsd_X2.append(block.get_unit_by_name('CA').coordinate)
    rmsd_X1, rmsd_X2 = np.array(rmsd_X1), np.array(rmsd_X2)
    rmsd_X1 = np.dot(rmsd_X1, Q) + t

    # calculate best matching
    if rmsd_X1.shape[0] > rmsd_X2.shape[0]:
        rmsd_X1, rmsd_X2 = rmsd_X2, rmsd_X1
    dist_mat = np.linalg.norm(rmsd_X1[:, None] - rmsd_X2[None, :], axis=-1)
    m = Munkres()
    cost_mat = dist_mat.copy()
    indexes = m.compute(cost_mat) # minimize cost
    dist = 0
    for i, j in indexes:
        dist += dist_mat[i][j] / len(indexes)
    return dist