import os
import math
import networkx as nx
import functools
import scipy.stats
import random
import sys
import copy
import numpy as np
import random
import json
from collections import defaultdict, Counter

import torch

import utils
sys.path.append('../../build')
import MatterSim


class EnvOracle(object):
    '''
        Environment oracle has access to environment graphs
    '''

    def __init__(self, scan_file):
        self.scans = set()
        self.graph = {}
        self.paths = {}
        self.distances = {}

        scans = set(open(scan_file).read().strip().split('\n'))
        self.add_scans(scans)

    def add_scans(self, scans, path=None):
        new_scans = set.difference(scans, self.scans)
        if new_scans:
            print('Loading navigation graphs for %d scans' % len(new_scans))
            for scan in new_scans:
                graph, paths, distances = self._compute_shortest_paths(scan, path=path)
                self.graph[scan] = graph
                self.paths[scan] = paths
                self.distances[scan] = distances
            self.scans.update(new_scans)

    def _compute_shortest_paths(self, scan, path=None):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        graph = utils.load_nav_graphs(scan, path=path)
        paths = dict(nx.all_pairs_dijkstra_path(graph))
        distances = dict(nx.all_pairs_dijkstra_path_length(graph))
        return graph, paths, distances

    def find_nearest_point(self, scan, start_point, end_points):
        result = (1e9, None)
        for end_point in end_points:
            d = self.distances[scan][start_point][end_point]
            if d < result[0]:
                result = (d, end_point)
        return result

    def find_nearest_point_on_a_path(self, scan, current_point, start_point,
            end_point):
        path = self.paths[scan][start_point][end_point]
        return self.find_nearest_point(scan, current_point, path)

    def distance_between_two_sets_of_nodes(self, scan, set_a, set_b):
        result = (1e9, None, None)
        for x in set_a:
            d, y = self.find_nearest_point(scan, x, set_b)
            if d < result[0]:
                result = (d, x, y)
        return result

    def get_graph(self, scan):
        return self.graph[scan]

    def get_path(self, scan, start_point, end_point):
        return self.paths[scan][start_point][end_point]

    def get_distance(self, scan, start_point, end_point):
        return self.distances[scan][start_point][end_point]

    def get_neighbors(self, scan, point):
        return self.graph[scan].neighbors(point)


class NavTeacher(object):
    '''
        Curiosity-Encouraging navigation teacher output:
            - Reference action
            - Actions that are mistaken in the past while executing the same language instruction
    '''

    def __init__(self, env_oracle):
        self.env_oracle = env_oracle

    def _shortest_path_action(self, ob):

        if ob['ended']:
            return -1

        scan = ob['scan']
        start_point = ob['viewpoint']

        _, target_point = self.env_oracle.find_nearest_point(
            scan, start_point, ob['target_viewpoints'])

        if start_point == target_point:
            return 0

        path = self.env_oracle.get_path(scan, start_point, target_point)
        next_point = path[1]
        for i, loc_attr in enumerate(ob['adj_loc_list']):
            if loc_attr['nextViewpointId'] == next_point:
                return i

        # Next nextViewpointId not found! This should not happen!
        print('adj_loc_list:', adj_loc_list)
        print('next point:', next_point)
        long_id = '{}_{}'.format(scan, start_point)
        print('long Id:', long_id)
        raise Exception('Bug: next viewpoint not in adj_loc_list')

    def __call__(self, obs):
        return list(map(self._shortest_path_action, obs))

    def _neg_actions(self, idx, info_list):
        neg_targets = []
        bad_next_viewpoints = defaultdict(set)

        for info in info_list[:-1]:
            neg_targets.append([])

            # If episode is over, no negative actions are added
            if info['nav_target'] == -1:
                continue

            ob = info['ob']
            scan = ob['scan']
            long_id = '_'.join([ob['viewpoint'], ob['subgoal_instr_id']])

            next_viewpoints = [
                loc['nextViewpointId'] for loc in ob['adj_loc_list']]

            for viewpoint in bad_next_viewpoints[long_id]:
                neg_targets[-1].append(
                    next_viewpoints.index(viewpoint) + idx * info['num_a'])

            # Add to set viewpoint of the non-optimal action
            if info['nav_a'] != info['nav_target']:
                viewpoint = ob['adj_loc_list'][info['nav_a']]['nextViewpointId']
                bad_next_viewpoints[long_id].add(viewpoint)

        return neg_targets

    def all_neg_nav(self, batch_info_list):
        neg_target_lists = map(self._neg_actions, range(len(batch_info_list)),
            batch_info_list)

        neg_targets = []
        neg_offsets = []
        for pos in zip(*neg_target_lists):
            neg_offset = []
            neg_target = []
            l = 0
            for item in pos:
                neg_target.extend(item)
                neg_offset.append(l)
                l += len(item)
            neg_targets.append(np.array(neg_target, dtype=np.int64))
            neg_offsets.append(neg_offset)

        return neg_targets, np.array(neg_offsets, dtype=np.int64)


class AskTeacher(object):
    '''
        Help-request teacher suggests:
            - Whether the agent should request help
            - Reasons for requesting (lost, uncertain_wrong, never_asked)
    '''

    reason_labels = ['lost', 'uncertain_wrong', 'already_asked']

    def __init__(self, hparams, agent_ask_actions, env_oracle, anna):

        self.uncertain_threshold = hparams.uncertain_threshold
        self.env_oracle = env_oracle
        self.anna = anna

        self.DO_NOTHING   = agent_ask_actions.index('do_nothing')
        self.REQUEST_HELP = agent_ask_actions.index('request_help')
        self.IGNORE       = -1

        self.LOST = self.reason_labels.index('lost')
        self.UNCERTAIN_WRONG = self.reason_labels.index('uncertain_wrong')
        self.ALREADY_ASKED = self.reason_labels.index('already_asked')

        self.no_ask = self.ask_every = self.random_ask = 0
        if hparams.ask_baseline is not None:
            if 'no_ask' in hparams.ask_baseline:
                self.no_ask = 1
            if 'ask_every' in hparams.ask_baseline:
                self.ask_every = int(hparams.ask_baseline.split(',')[-1])
            if 'random_ask' in hparams.ask_baseline:
                self.random_ask = float(hparams.ask_baseline.split(',')[-1])

    def _next_ask(self, idx, ob):

        if self.no_ask:
            return self.IGNORE, 'no_ask'

        if ob['ended']:
            return self.IGNORE, 'ended'

        scan = ob['scan']
        viewpoint = ob['viewpoint']
        if not self.anna.can_request(scan, viewpoint):
            return self.IGNORE, 'no_route'

        if self.random_ask:
            if random.random() < self.random_ask:
                return self.REQUEST_HELP, 'random_ask'

        # Baseline: ask every k steps
        if self.ask_every:
            if ob['time'] - ob['last_ask'] >= self.ask_every:
                return self.REQUEST_HELP, 'every_%d' % self.ask_every

        return self.DO_NOTHING, 'pass'

    def _should_ask(self, idx, info_list):

        scan = info_list[0]['ob']['scan']
        seq_len = len(info_list) - 1

        ask_targets = [self.DO_NOTHING] * seq_len
        ask_reason_targets = [
            ([0] * len(self.reason_labels)) for _ in range(seq_len)]
        ask_reasons = [[] for _ in range(seq_len)]

        # Can't request
        for i in range(seq_len):

            ob = info_list[i]['ob']
            if ob['ended']:
                ask_targets[i] = self.IGNORE
                ask_reasons[i].append('ended')

            viewpoint = ob['viewpoint']
            if not self.anna.can_request(scan, viewpoint):
                ask_targets[i] = self.IGNORE
                ask_reasons[i].append('no_route')

        # Request due to being lost
        last_pos = -1
        for i, info in enumerate(info_list):
            # Split trajectory into segments during each of which the agent is
            # on a task
            # A segment ends when
            # (a) The episode terminates
            # (b) The agent asks in the current step
            # (c) The agent stops at the current step
            if i == seq_len or \
                info['ask_a'] == self.REQUEST_HELP or \
                info['nav_a'] == 0:

                goals = info['ob']['target_viewpoints']
                min_d = 1e9
                lost_step = i + 1
                for j in range(i, last_pos, -1):

                    assert info_list[j]['ob']['target_viewpoints'] == goals

                    curr_point = info_list[j]['ob']['viewpoint']

                    d, _ = self.env_oracle.find_nearest_point(
                        scan, curr_point, goals)

                    if d <= min_d:
                        min_d = d
                        lost_step = j + 1

                for j in range(lost_step, min(seq_len, i + 1)):
                    #if ask_targets[j] == self.DO_NOTHING:
                    if ask_targets[j] != self.IGNORE:
                        ask_targets[j] = self.REQUEST_HELP
                        ask_reason_targets[j][self.LOST] = 1
                        ask_reasons[j].append('lost')

                last_pos = i

            if info['ob']['ended']:
                break

        # Request due to being uncertain and wrong!
        for i, info in enumerate(info_list[:-1]):
            if ask_targets[i] == self.IGNORE:
                continue
            nav_dist = info['nav_dist']
            ob = info['ob']
            entropy = scipy.stats.entropy(nav_dist, base=len(nav_dist))
            if entropy >= self.uncertain_threshold and \
                info['nav_argmax'] != info['nav_target']:
                ask_targets[i] = self.REQUEST_HELP
                ask_reason_targets[i][self.UNCERTAIN_WRONG] = 1
                ask_reasons[i].append('uncertain_wrong')

        # NOT request due to previously requested at the current location
        ask_points = set()
        for i, info in enumerate(info_list[:-1]):
            if ask_targets[i] == self.IGNORE:
                continue
            viewpoint = info['ob']['viewpoint']
            if viewpoint in ask_points:
                ask_targets[i] = self.DO_NOTHING
                ask_reason_targets[i][self.ALREADY_ASKED] = 1
                ask_reasons[i].append('already_asked')
            if info['ask_a'] == self.REQUEST_HELP:
                ask_points.add(viewpoint)

        for i in range(seq_len):
            if not ask_reasons[i]:
                ask_reasons[i].append('pass')

        return ask_targets, ask_reason_targets, ask_reasons

    def __call__(self, batch_info_list):
        targets, reason_targets, reasons = zip(*list(map(
            self._should_ask, range(len(batch_info_list)), batch_info_list)))
        targets = np.array(targets, dtype=np.int64)
        reason_targets = np.array(reason_targets, dtype=np.int64)

        return targets, reason_targets, reasons

    def next_ask(self, obs):
        target, reason = zip(*list(map(self._next_ask, range(len(obs)), obs)))
        return list(target), list(reason)


class Teacher(object):

    def __init__(self, hparams, agent_ask_actions, env_oracle, anna):

        self.nav_oracle = make_oracle('nav', env_oracle)
        self.ask_oracle = make_oracle('ask', hparams, agent_ask_actions,
            env_oracle, anna)

    def next_ask(self, obs):
        return self.ask_oracle.next_ask(obs)

    def all_ask(self, batch_info_list):
        return self.ask_oracle(batch_info_list)

    def next_nav(self, obs):
        return self.nav_oracle(obs)

    def all_neg_nav(self, batch_info_list):
        return self.nav_oracle.all_neg_nav(batch_info_list)


class ANNA(object):
    '''
        Automatic Natural Navigation Assistant
    '''

    def __init__(self, hparams, env_oracle):

        self.env_oracle = env_oracle

        with open(hparams.anna_routes_path) as f:
            data = json.load(f)

        self.routes = defaultdict(dict)
        for scan, routes in data.items():
            for r in routes:
                start_point = r['path'][0]
                if start_point not in self.routes[scan]:
                    self.routes[scan][start_point] = [r]
                else:
                    self.routes[scan][start_point].append(r)

        # Pre-compute zones of attention
        radius = hparams.start_point_radius
        self.requestable_points = defaultdict(lambda: defaultdict(list))
        for scan in data:
            for v in self.env_oracle.get_graph(scan):
                if v in self.routes[scan]:
                    self.requestable_points[scan][v].append(v)
                for u in self.env_oracle.get_neighbors(scan, v):
                    if self.env_oracle.get_distance(scan, v, u) <= radius and \
                        u in self.routes[scan]:
                        self.requestable_points[scan][v].append(u)

        self.random = random
        self.random.seed(hparams.seed)

        self.cached_results = defaultdict(dict)
        self.split_name = None
        self.is_eval = None
        #self.hparams = hparams

    def can_request(self, scan, viewpoint):
        return bool(self.requestable_points[scan][viewpoint])

    def get_result(self, results):
        result = results[0] if self.is_eval else self.random.choice(results)
        """
        if self.hparams.instruction_baseline == 'language_only':
            try:
                instruction = result['instruction']
                result['instruction'] = instruction[:instruction.index('.') + 1].rstrip()
            except ValueError:
                pass
        """
        return result

    def __call__(self, ob):

        scan = ob['scan']
        viewpoint = ob['viewpoint']
        goal_viewpoints = ob['goal_viewpoints']

        query_id = '_'.join([scan, viewpoint] + sorted(goal_viewpoints))

        cache = self.cached_results[self.split_name]

        if query_id in cache:
            results = cache[query_id]
            return self.get_result(results)

        valid_viewpoints = self.requestable_points[scan][viewpoint]

        # Gather all valid routes
        valid_routes = []
        for v in valid_viewpoints:
            valid_routes.extend(self.routes[scan][v])

        # Find departure node and goal nearest to depart node
        distances, depart_nodes, nearest_goals = zip(*[
            self.env_oracle.distance_between_two_sets_of_nodes(
                scan, r['path'], goal_viewpoints) for r in valid_routes])

        best_d = min(distances)

        results = [{
                'path_id' : r['path_id'],
                'request_node': viewpoint,
                'instruction' : r['instruction'],
                'view_id': r['view_id'],
                'start_node': r['path'][0],
                'depart_node' : v,
                'goal_node': g
            }
            for d, v, g, r in \
                zip(distances, depart_nodes, nearest_goals, valid_routes)
                    if abs(best_d - d) < 1e-6 and r['split'] == self.split_name]

        cache[query_id] = results

        return self.get_result(results)


def make_oracle(oracle_type, *args, **kwargs):

    if oracle_type == 'env_oracle':
        return EnvOracle(*args, **kwargs)

    if oracle_type == 'teacher':
        return Teacher(*args, **kwargs)

    if oracle_type == 'nav':
        return NavTeacher(*args, **kwargs)

    if oracle_type == 'ask':
        return AskTeacher(*args, **kwargs)

    if oracle_type == 'anna':
        return ANNA(*args, **kwargs)

    return None



