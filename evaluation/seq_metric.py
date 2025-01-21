#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import sqrt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from Bio.Align import substitution_matrices, PairwiseAligner

from utils.const import aa_smiles


def aar(candidate, reference):
    hit = 0
    for a, b in zip(candidate, reference):
        if a == b:
            hit += 1
    return hit / len(reference)


# 3. AAR based on fingerprint similarity
SIM_MAT = {}

def _init_sim_mat():
    global SIM_MAT
    mols = [Chem.MolFromSmiles(aa_smiles[aa]) for aa in aa_smiles]
    fps = [AllChem.GetMorganFingerprint(mol, 2) for mol in mols]
    for i, aa1 in enumerate(aa_smiles):
        for j, aa2 in enumerate(aa_smiles):
            SIM_MAT[aa1 + aa2] = SIM_MAT[aa2 + aa1] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
    max_sim, min_sim = max(SIM_MAT.values()), min(SIM_MAT.values())
    for pair in SIM_MAT:
        SIM_MAT[pair] = (SIM_MAT[pair] - min_sim) / (max_sim - min_sim)
    # print(f'Random AA pair similarity: {sum(SIM_MAT.values()) / len(SIM_MAT)}')


_init_sim_mat()


def sim_aar(candidate, reference):
    score = 0
    for x, y in zip(candidate, reference):
        score += SIM_MAT.get(x + y, 0)
    score = score / len(reference)
    return score


def slide_aar(candidate, reference, aar_func):
    '''
    e.g.
     candidate: AILPV
     reference: ILPVH

     should be matched as
     AILPV
      ILPVH

    To do this, we slide the candidate and calculate the maximum aar:
        A
       AI
      AIL
     AILP
    AILPV
    ILPV 
    LPV  
    PV   
    V    
    '''
    special_token = ' '
    ref_len = len(reference)
    padded_candidate = special_token * (ref_len - 1) + candidate + special_token * (ref_len - 1)
    value = 0
    for start in range(len(padded_candidate) - ref_len + 1):
        value = max(value, aar_func(padded_candidate[start:start + ref_len], reference))
    return value


def align_sequences(sequence_A, sequence_B, **kwargs):
    """
    Performs a global pairwise alignment between two sequences
    using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
    as implemented in Biopython. Returns the alignment, the sequence
    identity and the residue mapping between both original sequences.
    """

    sub_matrice = substitution_matrices.load('BLOSUM62')
    aligner = PairwiseAligner()
    aligner.substitution_matrix = sub_matrice
    if kwargs.get('local', False):
        aligner.mode = 'local'
    alns = aligner.align(sequence_A, sequence_B)

    best_aln = alns[0]
    aligned_A, aligned_B = best_aln

    base = sqrt(aligner.score(sequence_A, sequence_A) * aligner.score(sequence_B, sequence_B))
    seq_id = aligner.score(sequence_A, sequence_B) / base
    return (aligned_A, aligned_B), seq_id


if __name__ == '__main__':
    print(align_sequences('PKGYAAPSA', 'KPAVYKFTL'))
    print(align_sequences('KPAVYKFTL', 'PKGYAAPSA'))
    print(align_sequences('PKGYAAPSA', 'PKGYAAPSA'))
    print(align_sequences('KPAVYKFTL', 'KPAVYKFTL'))