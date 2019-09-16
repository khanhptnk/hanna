import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F


class BaseAgent(object):

    def __init__(self):
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
        self.results_path = None

    def write_results(self, traj):
        output = []
        for k, v in self.results.items():
            item = { 'instr_id' : k }
            item.update(v)
            output.append(item)

        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()

        self.results = {}
        looped = False
        traj = []
        with torch.no_grad():
            while True:
                for t in self.rollout():
                    if t['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[t['instr_id']] = t
                        traj.append(t)
                if looped:
                    break
        return traj

    def add_is_success(self, is_success):
        for instr_id, s in is_success:
            self.results[instr_id]['is_success'] = s


