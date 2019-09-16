import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets, load_nav_graphs
from ask_agent import AskAgent

class Evaluation(object):

    def __init__(self, hparams, splits, data_path):

        self.success_radius = hparams.success_radius
        self.splits = splits

        self.scans = set()
        self.graphs = {}
        self.distances = {}

        if splits:
            self.load_data(load_datasets(splits, data_path,
                prefix=hparams.data_prefix))

        self.hparams = hparams

    def load_data(self, data):
        self.gt = {}
        self.instr_ids = []
        scans = []
        for item in data:
            self.gt[str(item['path_id'])] = item
            if isinstance(item['path_id'], int):
                self.instr_ids.extend(['%d_%d' % (item['path_id'],i)
                    for i in range(len(item['instructions']))])
            else:
                self.instr_ids.extend(['%s_%d' % (item['path_id'],i)
                    for i in range(len(item['instructions']))])
            scans.append(item['scan'])
        self.instr_ids = set(self.instr_ids)
        scans = set(scans)

        new_scans = set.difference(scans, self.scans)
        if new_scans:
            for scan in new_scans:
                self.graphs[scan] = load_nav_graphs(scan)
                self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(self.graphs[scan]))
        self.scans.update(new_scans)

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _score_item(self, item):

        instr_id = item['instr_id']
        path = [p[0] for p in item['agent_pose']]

        gt = self.gt[instr_id[:instr_id.rfind('_')]]
        scan = gt['scan']

        self.scores['instr_ids'].append(instr_id)
        self.scores['steps'].append(len(path) - 1)

        error = oracle_error = 1e9
        shortest_distance = 1e9
        oracle_stop = None
        for shortest_path in gt['paths']:
            start = shortest_path[0]
            assert start == path[0], 'Result trajectories should include the start position'
            goal = shortest_path[-1]
            final_pos = path[-1]

            if self.distances[scan][final_pos][goal] <= error:
                error = self.distances[scan][final_pos][goal]
                shortest_distance = self.distances[scan][start][goal]

            nearest_pos = self._get_nearest(scan, goal, path)
            this_oracle_error = self.distances[scan][nearest_pos][goal]
            if this_oracle_error < oracle_error:
                oracle_error = this_oracle_error
                oracle_stop  = nearest_pos

        self.scores['errors'].append(error)
        self.scores['oracle_errors'].append(oracle_error)
        self.scores['optimal_lengths'].append(shortest_distance)

        distance = 0
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[scan][prev][curr]
            prev = curr
        self.scores['lengths'].append(distance)

        path_oracle_stop = path[:path.index(oracle_stop) + 1]
        distance = 0
        prev = path_oracle_stop[0]
        for curr in path_oracle_stop[1:]:
            distance += self.distances[scan][prev][curr]
            prev = curr
        self.scores['oracle_lengths'].append(distance)

        if 'agent_ask' not in item:
            return

        nav_actions = item['agent_nav']
        ask_actions = item['agent_ask']
        target_viewpoints = item['target_viewpoints']

        assert len(nav_actions) == len(ask_actions) == \
            len(target_viewpoints) - 1 == len(path) - 1

        asked = False
        for nav_a, ask_a, v, targets in \
                zip(nav_actions, ask_actions, path[1:], target_viewpoints[:-1]):

            if asked and nav_a <= 0:
                assert len(targets) == 1
                g = targets[0]
                self.scores['target_errors'].append(self.distances[scan][v][g])
                asked = False

            if ask_a == AskAgent.ask_actions.index('request_help'):
                asked = True

        if asked:
            v = path[-1]
            g = target_viewpoints[-1][0]
            self.scores['target_errors'].append(self.distances[scan][v][g])

    def is_success(self, d):
        return d <= self.success_radius

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item)
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        assert len(self.scores['errors']) == len(self.instr_ids)

        score_summary = {
            'error': np.average(self.scores['errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['steps']),
            'length': np.average(self.scores['lengths'])
        }
        is_success = [self.is_success(d) for d in self.scores['errors']]
        score_summary['success_rate'] = np.average(is_success)
        score_summary['spl'] = np.average([s * l / max(p, l) for s, p, l in
            zip(is_success, self.scores['lengths'], self.scores['optimal_lengths'])])

        is_success_oracle = [self.is_success(d) for d in self.scores['oracle_errors']]
        score_summary['oracle_success_rate'] = np.average(is_success_oracle)
        score_summary['oracle_spl'] = np.average([s * l / max(p, l) for s, p, l in
            zip(is_success, self.scores['oracle_lengths'], self.scores['optimal_lengths'])])

        if self.scores['target_errors']:
            is_success_target = [self.is_success(d) for d in self.scores['target_errors']]
            score_summary['target_success_rate'] = np.average(is_success_target)
        else:
            score_summary['target_success_rate'] = -1

        return score_summary, self.scores, \
            list(zip(self.scores['instr_ids'], is_success))





