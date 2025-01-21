#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RelaxTask:
    in_path: str
    current_path: str
    info: dict
    status: str
    rec_chain: str
    pep_chain: str
    rfdiff_relax: bool = False
    dG: Optional[float] = None

    def set_dG(self, dG):
        self.dG = dG

    def get_in_path_with_tag(self, tag):
        name, ext = os.path.splitext(self.in_path)
        new_path = f'{name}_{tag}{ext}'
        return new_path

    def set_current_path_tag(self, tag):
        new_path = self.get_in_path_with_tag(tag)
        self.current_path = new_path
        return new_path

    def check_current_path_exists(self):
        ok = os.path.exists(self.current_path)
        if not ok:
            self.mark_failure()
        elif os.path.getsize(self.current_path) == 0:
            ok = False
            self.mark_failure()
            os.unlink(self.current_path)
        return ok

    def update_if_finished(self, tag):
        out_path = self.get_in_path_with_tag(tag)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            # print('Already finished', out_path)
            self.set_current_path_tag(tag)
            self.mark_success()
            return True
        return False

    def can_proceed(self):
        self.check_current_path_exists()
        return self.status != 'failed'

    def mark_success(self):
        self.status = 'success'

    def mark_failure(self):
        self.status = 'failed'
    
    def mark_proceeding(self):
        self.status = 'proceeding'


class TaskScanner:

    def __init__(self, results, n_sample, rfdiff_relax):
        super().__init__()
        self.results = results
        self.n_sample = n_sample
        self.rfdiff_relax = rfdiff_relax
        self.visited = set()

    def scan(self) -> List[RelaxTask]: 
        tasks = []
        root_dir = os.path.dirname(self.results)
        with open(self.results, 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            item = json.loads(line)
            if item['number'] >= self.n_sample:
                continue
            _id = f"{item['id']}_{item['number']}"
            if _id in self.visited:
                continue
            gen_pdb = os.path.split(item['gen_pdb'])[-1]
            subdir = gen_pdb.split('_')[0]
            gen_pdb = os.path.join(root_dir, 'candidates', subdir, gen_pdb)
            tasks.append(RelaxTask(
                in_path=gen_pdb,
                current_path=gen_pdb,
                info=item,
                status='created',
                rec_chain=item['rec_chain'],
                pep_chain=item['lig_chain'],
                rfdiff_relax=self.rfdiff_relax
            ))
            self.visited.add(_id)
        return tasks