#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import argparse

import ray
import torch

from .base import TaskScanner, NoCYSFilter

from utils.logger import print_log

from .foldx_dG import run_openmm_relax


@ray.remote
def pipeline_pyrosetta(task):
    if torch.cuda.is_available():
        run_openmm_relax_remote = ray.remote(num_gpus=1/8, num_cpus=1)(run_openmm_relax)
    else:
        run_openmm_relax_remote = ray.remote(num_cpus=1)(run_openmm_relax)
    funcs = [
        run_openmm_relax_remote,
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
    parser.add_argument('--server_mode', action='store_true', help='Keep scanning as a server')
    return parser.parse_args()


def main(args):

    if args.n_cpus > 0:
        ray.init(num_cpus=args.n_cpus)
    else:
        ray.init() # use all available cpus
    scanner = TaskScanner(
        args.root_dir,
        filters=[NoCYSFilter()] if args.no_cys_filter else [],
        specify_result_dir=args.result_dir)

    while True:
        tasks = scanner.scan()
        if (len(tasks) == 0) and (not args.server_mode): break
        futures = [pipeline_pyrosetta.remote(t) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            for done_id in done_ids:
                done_task = ray.get(done_id)
                print_log(f'Remaining {len(futures)}. Finished {done_task.current_path}')
        time.sleep(1.0)


if __name__ == '__main__':
    main(parse())