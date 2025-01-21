#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil

from globals import FOLDX_BIN, CACHE_DIR


def foldx_minimize_energy(pdb_path, out_path):
    filename = os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
    tmpfile = os.path.join(CACHE_DIR, filename)
    shutil.copyfile(pdb_path, tmpfile)
    p = os.popen(f'cd {CACHE_DIR}; {FOLDX_BIN} --command=Optimize --pdb={filename}')
    p.read()
    p.close()
    os.remove(tmpfile)
    filename = 'Optimized_' + filename
    tmpfile = os.path.join(CACHE_DIR, filename)
    shutil.copyfile(tmpfile, out_path)
    os.remove(tmpfile)


def foldx_dg(pdb_path, rec_chains, lig_chains):
    filename = os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
    tmpfile = os.path.join(CACHE_DIR, filename)
    shutil.copyfile(pdb_path, tmpfile)
    rec, lig = ''.join(rec_chains), ''.join(lig_chains)
    p = os.popen(f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}')
    aff = float(p.read().split('\n')[-8].split(' ')[-1])
    p.close()
    os.remove(tmpfile)
    return aff
