#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass


@dataclass
class Task:
    item_id: str
    in_path: str
    current_path: str
    info: dict
    status: str
    rec_chains: List[str]
    pep_chain: str
    pep_seq: str
    root_dir: str
    out_data: Optional[Any] = None
    log: Optional[str] = None

    def set_log(self, log):
        self.log = log

    def set_data(self, data):
        self.out_data = data

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
            self.set_current_path_tag(tag)
            self.mark_proceeding()
            return True
        return False

    def can_proceed(self):
        self.check_current_path_exists()
        return self.status != 'failed'

    def mark_resubmit(self):
        self.status = 'resubmit'

    def mark_proceeding(self):
        self.status = 'proceeding'

    def mark_success(self):
        self.status = 'success'

    def mark_failure(self):
        self.status = 'failed'

    def is_resubmit(self):
        return self.status == 'resubmit'

    def is_success(self):
        return self.status == 'success'

'''
Filters: 0 for not passed, 1 for passed, 2 for waiting (prerequisites not satisfied)
'''
class BaseFilter:
    def check(self, task: Task):
        return 1
    

class NoCYSFilter(BaseFilter):
    def check(self, task: Task):
        return int('C' not in task.pep_seq)
    

class PyRosettaDGFilter(BaseFilter):

    def __init__(self, root_dir, th) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.th = th
    
    def check(self, task: Task):
        index_file = os.path.join(self.root_dir, 'results', 'pyrosetta_dG.txt')
        with open(index_file, 'r') as fin:
            for line in fin.readlines():
                if task.item_id in line:
                    dG = float(line.strip().split('\t')[1])
                    return int(dG < self.th)
        return 2
    

class ItemIDFilter(BaseFilter):
    def __init__(self, id_list) -> None:
        super().__init__()
        self.id_list = { _id: 1 for _id in id_list }

    def check(self, task: Task):
        return self.id_list.get(task.item_id, 0)


class TaskScanner:

    def __init__(self, root_dir, filters: List[BaseFilter]=[], specify_result_dir='results'):
        super().__init__()
        self.root_dir = root_dir
        self.visited = set()
        self.filters = filters
        self.specify_result_dir = specify_result_dir

    def scan(self) -> List[Task]: 
        tasks = []
        result_file = os.path.join(self.root_dir, self.specify_result_dir, 'results.jsonl')
        with open(result_file, 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            item = json.loads(line)
            pdb_id = item['id']
            _id = f"{pdb_id}_{item['number']}"
            if _id in self.visited:
                continue
            gen_pdb = os.path.split(item['gen_pdb'])[-1]
            gen_pdb = os.path.join(self.root_dir, self.specify_result_dir, pdb_id, gen_pdb)
            task = Task(
                item_id=_id,
                in_path=gen_pdb,
                current_path=gen_pdb,
                info=item,
                status='created',
                rec_chains=item['rec_chains'],
                pep_chain=item['lig_chain'],
                pep_seq=item['gen_seq'],
                root_dir=self.root_dir
            )
            check_status = 1
            for filter in self.filters:
                status = filter.check(task)
                if status == 0:
                    check_status = 0
                    break
                elif status == 1:
                    continue
                elif status == 2:
                    check_status = 2
                else:
                    raise ValueError(f'Filter status code {status} invalid')
            if check_status == 0:
                self.visited.add(_id)
            elif check_status == 1:
                tasks.append(task)
                self.visited.add(_id)
            elif check_status == 2:
                continue  # some criterion not ready yet
        return tasks