import os
import sys

import numpy as np
import json
import networkx as nx
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


from utils import load_nav_graphs

def success_rate(data):
    radius = 2
    succ = []
    for item in data:
        scan = item['scan']
        final_loc = item['agent_pose'][-1][0]
        succ.append(0)
        for target_loc in item['target_viewpoints'][0]:
            if distances[scan][final_loc][target_loc] <= radius:
                succ[-1] = 1
    return np.average(succ) * 100


def spl_score(data):
    radius = 2
    scores = []
    for item in data:
        scan = item['scan']
        start_loc = item['agent_pose'][0][0]
        agent_dist = 0
        prev_loc = start_loc
        for pose in item['agent_pose'][1:]:
            next_loc = pose[0]
            agent_dist += distances[scan][next_loc][prev_loc]
            prev_loc = next_loc

        final_loc = item['agent_pose'][-1][0]
        close_target_loc = None
        dist = 1e9
        for target_loc in item['target_viewpoints'][0]:
            if distances[scan][final_loc][target_loc] <= dist:
                dist = distances[scan][final_loc][target_loc]
                close_target_loc = target_loc

        if dist <= radius:
            opt_dist = distances[scan][start_loc][close_target_loc]
            scores.append(opt_dist / max(agent_dist, opt_dist))
        else:
            scores.append(0)

    return np.average(scores) * 100


def navigation_error(data):
    close_dist = []
    for item in data:
        scan = item['scan']
        final_loc = item['agent_pose'][-1][0]
        dist = 1e9
        for target_loc in item['target_viewpoints'][0]:
            dist = min(dist, distances[scan][final_loc][target_loc])
        close_dist.append(dist)
    return np.average(close_dist)


def num_requests(data):
    req = []
    for item in data:
        cnt = sum(x == 1 for x in item['agent_ask'])
        req.append(cnt)
    return np.average(req)


def nav_mistake_repeat(data):
    rep = []
    for item in data:
        opt_action = {}
        wrong_actions = defaultdict(set)
        for i, (pose, instr, a_true, a_pred, adj_list) in enumerate(zip(item['agent_pose'], item['instruction'], item['teacher_nav'], item['agent_nav'], item['adj_loc_list'])):

            pred_point = adj_list[a_pred]['nextViewpointId']
            best_point = adj_list[a_true]['nextViewpointId']

            rep.append(0)
            if a_true != -1:

                key = pose[0] + ' ' + instr + ' ' + ' '.join(sorted(item['target_viewpoints'][i]))

                if key in opt_action:
                    assert opt_action[key] == best_point
                else:
                    opt_action[key] = best_point

                if key in wrong_actions:
                    rep[-1] += pred_point in wrong_actions[key]
                if pred_point != best_point:
                    wrong_actions[key].add(pred_point)

    return np.average(rep) * 100


def ask_repeat(data):
    rep = []
    for item in data:
        ask_actions = defaultdict(set)
        for i, (pose, instr, a_pred, a_true) in enumerate(zip(item['agent_pose'], item['instruction'], item['agent_ask'], item['teacher_ask'])):
            point = pose[0]
            if a_pred == 1:
                assert a_true != -1
                key = pose[0] + ' ' + instr + ' ' + ' '.join(sorted(item['target_viewpoints'][i]))
                rep.append(0)
                if key in ask_actions:
                    rep[-1] += 1 in ask_actions[key]
                ask_actions[key].add(1)

    return np.average(rep) * 100


def reason_metrics(data, reason):
    pred = []
    true = []
    for item in data:
        for a_true, pred_reason, true_reason in zip(item['teacher_ask'], item['agent_reason'], item['teacher_reason']):
            if a_true != -1:
                pred.append(reason in pred_reason)
                true.append(reason in true_reason)

    return accuracy_score(true, pred) * 100, \
           precision_score(true, pred) * 100, \
           recall_score(true, pred) * 100, \
           f1_score(true, pred) * 100


def ask_dist(data):
    ask = []
    for item in data:
        for a_ask, a_true in zip(item['agent_ask'], item['teacher_ask']):
            if a_true != -1:
                ask.append(a_ask)
    return np.average(ask) * 100


def pred_reason_dist(data):
    total = 0
    cnt = defaultdict(int)
    for item in data:
        for reason, a_true in zip(item['agent_reason'], item['teacher_ask']):
            if a_true != -1:
                total += 1
                for x in reason:
                    cnt[x] += 1
    return cnt['lost'] / total * 100, cnt['uncertain_wrong'] / total * 100, \
           cnt['already_asked'] / total * 100

def true_reason_dist(data):
    total = 0
    cnt = defaultdict(int)
    for item in data:
        for reason, a_true in zip(item['teacher_reason'], item['teacher_ask']):
            if a_true != -1:
                total += 1
                for x in reason:
                    cnt[x] += 1
    return cnt['lost'] / total * 100, cnt['uncertain_wrong'] / total * 100, \
           cnt['already_asked'] / total * 100



data_path = '../../../data'

graphs = {}
distances = {}

with open(os.path.join(data_path, 'hanna/scan_split.json')) as f:
    scan_split = json.load(f)

for split in ['train', 'val', 'test']:
    for scan in scan_split[split]:
        graphs[scan] = load_nav_graphs(scan)
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(graphs[scan]))

filename = sys.argv[1]

with open(filename) as f:
    data = json.load(f)


print('Success rate: %.2f' % success_rate(data))
print('SPL: %.2f' % spl_score(data))
print('Navigation error: %.2f' % navigation_error(data))
print('Requests: %.1f' % num_requests(data))
print('Nav mistake repeat: %.2f' % nav_mistake_repeat(data))
print('Ask repeat: %.2f' % ask_repeat(data))
for reason in ['lost', 'uncertain_wrong', 'already_asked']:
    print_values = (reason,) + reason_metrics(data, reason)
    print('%s metrics (A|P|R|F1): %.1f,%.1f,%.1f,%.1f' % print_values)
print('Ask distribution: request %.0f nothing %.0f' % (ask_dist(data), 100 - ask_dist(data)))

print('Pred Reason distribution: lost %.1f  uncertain_wrong %.1f  already_asked %.1f ' % pred_reason_dist(data))
print('True Reason distribution: lost %.1f  uncertain_wrong %.1f  already_asked %.1f ' % true_reason_dist(data))



