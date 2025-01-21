#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from Bio.Align import substitution_matrices
from munkres import Munkres

from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_cb_interface, blocks_to_cb_coords, dist_matrix_from_blocks, blocks_interface


def _similarity_matrix(gen_mat: np.ndarray, ref_mat: np.ndarray, criterion: str='pearson'):
    '''
    calculating cosine similarity over embeddings of each residue
    Args:
        gen_mat: [Nag, Ngen]
        ref_mat: [Nag, Nref]
        criterion: cosine, pearson or spearman
    '''
    if criterion == 'cosine':
        d = gen_mat.T @ ref_mat  # [Ngen, Nref]
        norm_gen = np.sqrt((gen_mat * gen_mat).sum(0, keepdims=True)) # [1, Ngen]
        norm_ref = np.sqrt((ref_mat * ref_mat).sum(0, keepdims=True)) # [1, Nref]
        sim = d / (norm_gen.T @ norm_ref)
    elif criterion == 'pearson':
        A, B = gen_mat, ref_mat
        # Get number of rows in either A or B
        N = B.shape[0]

        # Store columnw-wise in A and B, as they would be used at few places
        sA = A.sum(0)
        sB = B.sum(0)

        # Basically there are four parts in the formula. We would compute them one-by-one
        p1 = N * np.einsum('ij,ik->kj',A,B)
        p2 = sA * sB[:,None]
        p3 = N * ((B ** 2).sum(0)) - (sB ** 2)
        p4 = N * ((A ** 2).sum(0)) - (sA ** 2)

        # Finally compute Pearson Correlation Coefficient as 2D array 
        pcorr = ((p1 - p2) / np.sqrt(p4 * p3[:,None]))
        sim = pcorr.T
    else:
        raise ValueError(f'Criterion {criterion} not supported')
    return sim


def _extract_blocks(pdb, rec_chains, lig_chains):
    chain2blocks = pdb_to_list_blocks(pdb, selected_chains=rec_chains + lig_chains, dict_form=True)
    rec_blocks, lig_blocks = [], []
    for c in rec_chains: rec_blocks.extend(chain2blocks[c])
    for c in lig_chains: lig_blocks.extend(chain2blocks[c])
    return rec_blocks, lig_blocks


def _dist_feature(rec_blocks, lig_blocks):
    #return dist_matrix_from_blocks(rec_blocks, lig_blocks)
    rec_cb_coords = blocks_to_cb_coords(rec_blocks)
    lig_cb_coords = blocks_to_cb_coords(lig_blocks)
    dist = np.linalg.norm(rec_cb_coords[:, None] - lig_cb_coords[None, :], axis=-1)  # [Nrec, Nlig]
    return dist

def _cb_dist(block1, block2):
    x1, x2 = blocks_to_cb_coords([block1, block2])
    return np.linalg.norm(
        np.array(x1) - np.array(x2)
    )


def _matching_score(block1, block2):
    X1 = np.array([unit.coordinate for unit in block1.units])
    elements1 = [unit.get_element() for unit in block1.units]
    X2 = np.array([unit.coordinate for unit in block2.units])
    elements2 = [unit.get_element() for unit in block2.units]
    if X1.shape[0] > X2.shape[0]:
        X1, X2 = X2, X1
        elements1, elements2 = elements2, elements1
    mat = np.linalg.norm(X1[:, None] - X2[None, :], axis=-1) # N1 * N2
    # optimal align
    m = Munkres()
    cost_mat = mat.copy()
    indexes = m.compute(cost_mat) # minimize cost
    hit_cnt = 0
    for i, j in indexes:
        if mat[i][j] < 2.0:
            if elements1[i] == elements2[j] and elements1[i] != 'C':
                hit_cnt += 10
            else:
                hit_cnt + 1
    return hit_cnt    


def interface_hit(gen_pdb, ref_pdb, rec_chains, gen_lig_chains, ref_lig_chains, th=0.9, strict=False, return_idx=False):
    gen_rec_blocks, gen_lig_blocks = _extract_blocks(gen_pdb, rec_chains, gen_lig_chains)
    ref_rec_blocks, ref_lig_blocks = _extract_blocks(ref_pdb, rec_chains, ref_lig_chains)
    _, (pocket_idx, ref_lig_idx) = blocks_cb_interface(ref_rec_blocks, ref_lig_blocks, 10.0)
    #_, (pocket_idx, ref_lig_idx) = blocks_interface(ref_rec_blocks, ref_lig_blocks, 5.0)
    ref_lig_blocks = [ref_lig_blocks[i] for i in ref_lig_idx]
    gen_dist_feature = _dist_feature([gen_rec_blocks[i] for i in pocket_idx], gen_lig_blocks)
    ref_dist_feature = _dist_feature([ref_rec_blocks[i] for i in pocket_idx], ref_lig_blocks)
    
    similarity = _similarity_matrix(gen_dist_feature, ref_dist_feature) # [Ngen, Nref]
    hit_pairs = np.nonzero(similarity > th)

    # subscore
    sub_matrice = substitution_matrices.load('BLOSUM62')
    sub_scores = np.zeros_like(similarity)

    cnt = 0
    for i, j in zip(*hit_pairs):
        gen_block, ref_block = gen_lig_blocks[i], ref_lig_blocks[j]
        res1, res2 = VOCAB.abrv_to_symbol(gen_block.abrv), VOCAB.abrv_to_symbol(ref_block.abrv)
        if res1 == '?' or res2 == '?':
            continue
        score = sub_matrice[res1, res2]
        if strict:
            pair_match_score = _matching_score(gen_block, ref_block)
        else:
            pair_match_score = 1e10
        if score > 0 and pair_match_score > 1:
            cnt += 1
            sub_scores[i][j] = score
    if cnt == 0:
        return []

    # optimal align
    m = Munkres()
    cost_mat = -sub_scores.copy()
    transpose = False
    if cost_mat.shape[0] > cost_mat.shape[1]:
        cost_mat, transpose = cost_mat.T, True
    indexes = m.compute(cost_mat) # minimize cost
    pairs = []
    for i, j in indexes:
        if transpose:
            i, j = j, i
        gen_block, ref_block = gen_lig_blocks[i], ref_lig_blocks[j]
        score = sub_scores[i][j]
        if score > 0:
            if return_idx:
                pairs.append((gen_block, ref_block, similarity[i][j], score, i, j))
            else:
                pairs.append((gen_block, ref_block, similarity[i][j], score))

    # print(hit_pairs)
    # total_score = 0
    # for pair in pairs:
    #     total_score += pair[-1]
    # return [None for _ in range(int(total_score))]
    return pairs


if __name__ == '__main__':
    import sys
    pairs = interface_hit(sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4].split(','), sys.argv[5].split(','))
    print(len(pairs))
    for pair in pairs:
        print(pair)