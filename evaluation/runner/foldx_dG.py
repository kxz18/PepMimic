#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import argparse

import ray

from .base import Task, TaskScanner, NoCYSFilter

from utils.logger import print_log
from evaluation.dG.openmm_relaxer import ForceFieldMinimizer
from evaluation.dG.foldx_energy import foldx_minimize_energy, foldx_dg


@ray.remote(num_gpus=1/8, num_cpus=1)
def run_openmm_relax(task: Task):
    if not task.can_proceed():
        return task
    if task.update_if_finished('openmm'):
        return task

    in_path = task.current_path
    out_path =  task.set_current_path_tag('openmm')
    force_field = ForceFieldMinimizer()
    try:
        force_field(in_path, out_path)
        task.mark_proceeding()
    except Exception as e:
        task.set_log(str(e))
        task.mark_failure()
    return task


@ray.remote(num_cpus=1)
def run_foldx_relax(task: Task):
    if not task.can_proceed():
        return task
    if task.update_if_finished('foldx'):
        return task

    in_path = task.current_path
    out_path = task.set_current_path_tag('foldx')
    try:
        foldx_minimize_energy(in_path, out_path)
        task.mark_proceeding()
    except Exception as e:
        task.set_log(str(e))
        task.mark_failure()
    return task


@ray.remote(num_cpus=1)
def run_foldx_dg(task: Task):
    if not task.can_proceed():
        return task
    try:
        dG = foldx_dg(task.current_path, task.rec_chains, [task.pep_chain])
        task.set_data(dG)
        task.mark_success()
    except Exception as e:
        task.set_data(1e10)  # very large value
        task.set_log(str(e))
        task.mark_failure()
    return task
            

@ray.remote
def pipeline_pyrosetta(task):
    funcs = [
        run_openmm_relax,
        run_foldx_relax,
        run_foldx_dg
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)


def parse():
    parser = argparse.ArgumentParser(description='calculating dG using pyrosetta')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the target')
    parser.add_argument('--n_cpus', type=int, default=-1, help='Default using all cpus')
    parser.add_argument('--no_cys_filter', action='store_true')
    parser.add_argument('--result_dir', type=str, default='results', help='Result directory')
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
    all_results = sorted(all_results, key=lambda tup: tup[1])
    with open(out_file, 'w') as fout:
        for i, (_id, dG, seq) in enumerate(all_results):
            fout.write(f'{_id}\t{dG}\t{seq}\n')

def main(args):

    if args.n_cpus > 0:
        ray.init(num_cpus=args.n_cpus)
    else:
        ray.init() # use all available cpus
    scanner = TaskScanner(
        args.root_dir,
        filters=[NoCYSFilter()] if args.no_cys_filter else [],
        specify_result_dir=args.result_dir)

    out_file = os.path.join(args.root_dir, args.result_dir, 'foldx_dG.txt')
    if os.path.exists(out_file):
        os.remove(out_file)

    while True:
        tasks = scanner.scan()
        if len(tasks) == 0: break
        futures = [pipeline_pyrosetta.remote(t) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            new_results = []
            for done_id in done_ids:
                done_task = ray.get(done_id)
                if done_task.out_data is None:
                    done_task.out_data = 1e10
                print_log(f'Remaining {len(futures)}. Finished {done_task.current_path}, dG {done_task.out_data}')
                new_results.append((done_task.item_id, done_task.out_data, done_task.pep_seq))
            sort_write(out_file, new_results)
        time.sleep(1.0)


if __name__ == '__main__':
    main(parse())