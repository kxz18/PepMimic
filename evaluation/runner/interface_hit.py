#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import argparse

import ray

from .base import Task, TaskScanner, NoCYSFilter

from utils.logger import print_log
from evaluation.interface_hit import interface_hit


def _get_ref_info(root_dir):
    info = {}
    with open(os.path.join(root_dir, 'index.txt'), 'r') as fin:
        lines = fin.readlines()
    for line in lines:
        pdb_id, rec_chains, lig_chains, _ = line.strip().split('\t')
        info[pdb_id] = (
            os.path.join(root_dir, pdb_id + '.pdb'),
            rec_chains.split(','),
            lig_chains.split(',')
        )
    return info


@ray.remote(num_cpus=1)
def run_interface_hit(task: Task, ref_info: dict):
    if not task.can_proceed():
        return task
    pdb_id = '_'.join(task.item_id.split('_')[:-1])
    # if pdb_id == 'ranked':
    #     pdb_id = pdb_id + '_' + str(task.item_id.split('_')[1])
    ref_pdb, rec_chains, lig_chains = ref_info[pdb_id]

    pairs = interface_hit(task.current_path, ref_pdb, rec_chains, [task.pep_chain], lig_chains)
    task.set_data(pairs)
    task.mark_success()
    return task
            

@ray.remote
def pipeline_interface_hit(task):
    funcs = [
        run_interface_hit
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)


def parse():
    parser = argparse.ArgumentParser(description='calculating dG using pyrosetta')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the target')
    parser.add_argument('--specify_result_dir', type=str, default='results', help='Directory for candidates under the root directory')
    parser.add_argument('--no_cys_filter', action='store_true')
    parser.add_argument('--n_cpus', type=int, default=-1, help='Default using all cpus')
    parser.add_argument('--out_file', type=str, default=None)
    return parser.parse_args()


def sort_write(out_file, new_results):
    old_results = []
    if os.path.exists(out_file):
        with open(out_file, 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            _id, dG, seq = line.strip('\n').split('\t')
            old_results.append((_id, float(dG), seq))
    all_results = old_results + new_results
    all_results = sorted(all_results, key=lambda tup: tup[1], reverse=True)
    with open(out_file, 'w') as fout:
        for i, (_id, dG, seq) in enumerate(all_results):
            fout.write(f'{_id}\t{dG}\t{seq}\n')

def main(args):

    if args.n_cpus > 0:
        ray.init(num_cpus=args.n_cpus)
    else:
        ray.init() # use all available cpus
    scanner = TaskScanner(args.root_dir, filters=[NoCYSFilter()] if args.no_cys_filter else [], specify_result_dir=args.specify_result_dir)

    if args.out_file is None:
        out_file = os.path.join(args.root_dir, args.specify_result_dir, 'interface_hit.txt')
    else:
        out_file = args.out_file
    if os.path.exists(out_file):
        os.remove(out_file)
    ref_info = _get_ref_info(args.root_dir)

    while True:
        tasks = scanner.scan()
        if len(tasks) == 0: break
        futures = [run_interface_hit.remote(t, ref_info) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            new_results = []
            for done_id in done_ids:
                done_task = ray.get(done_id)
                print_log(f'Remaining {len(futures)}. Finished {done_task.current_path}, num pairs {len(done_task.out_data)}')
                new_results.append((done_task.item_id, len(done_task.out_data), done_task.pep_seq))
            sort_write(out_file, new_results)
        time.sleep(1.0)


if __name__ == '__main__':
    main(parse())
