import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import random
import os
import time
import math
import numpy as np
import pandas as pd
import argparse
import json
from collections import defaultdict, Counter
from argparse import Namespace
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

from utils import read_vocab, Tokenizer, timeSince
from env import Batch, ImageFeatures, Simulator
from model import AgentModel
from ask_agent import AskAgent, SimpleAgent
from verbal_ask_agent import VerbalAskAgent
from oracle import AskTeacher

from eval import Evaluation
from oracle import *
from flags import make_parser

def set_path():
    OUTPUT_DIR = os.getenv('PT_OUTPUT_DIR', 'output')

    hparams.exp_dir = os.path.join(OUTPUT_DIR, hparams.exp_name)
    if not os.path.exists(hparams.exp_dir):
        os.makedirs(hparams.exp_dir)

    hparams.load_path = hparams.load_path if hasattr(hparams, 'load_path') and \
        hparams.load_path is not None else \
        os.path.join(hparams.exp_dir, '%s_last.ckpt' % hparams.exp_name)

    DATA_DIR = os.getenv('PT_DATA_DIR', '../../../data')
    hparams.data_path = os.path.join(DATA_DIR, hparams.data_dir)
    hparams.img_features = os.path.join(
        DATA_DIR, 'img_features/ResNet-152-imagenet.tsv')

    hparams.anna_routes_path = os.path.join(hparams.data_path, 'routes.json')
    hparams.scan_path = os.path.join(DATA_DIR, 'connectivity/scans.txt')

def save(path, model, optimizer, iter, best_metrics, train_env):
    ckpt = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'hparams'         : hparams,
            'iter'            : iter,
            'best_metrics'    : best_metrics,
            'data_idx'        : train_env.ix,
            'vocab'           : train_env.tokenizer.vocab
        }
    torch.save(ckpt, path)

def load(path, device):
    global hparams
    ckpt = torch.load(path, map_location=device)
    hparams = ckpt['hparams']

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()

    return ckpt

def compute_ask_stats(agent, traj):

    queries_per_ep = []
    agent_ask = []
    teacher_ask = []

    agent_reasons = defaultdict(list)
    teacher_reasons = defaultdict(list)
    loss_str = ''
    ask_repeats = 0
    nav_repeats = 0
    total_nav = 0

    request_index = AskAgent.ask_actions.index('request_help')

    for i, t in enumerate(traj):
        num_queries = 0
        nav_action_dict = defaultdict(set)
        ask_action_dict = defaultdict(set)
        total_nav += len(t['agent_nav'])
        for i, a in enumerate(t['teacher_ask']):
            viewpoint = t['agent_pose'][i][0]
            instruction = t['instruction'][i]
            nav_action = t['agent_nav'][i]
            nav_teacher_action = t['teacher_nav'][i]
            target_viewpoints = t['target_viewpoints'][i]
            key = viewpoint + ' ' + instruction + ' ' + ' '.join(sorted(target_viewpoints))
            if nav_teacher_action != -1:
                pred_point = t['adj_loc_list'][i][nav_action]['nextViewpointId']
                best_point = t['adj_loc_list'][i][nav_teacher_action]['nextViewpointId']
                if key in nav_action_dict:
                    nav_repeats += pred_point in nav_action_dict[key]
                if pred_point != best_point:
                    nav_action_dict[key].add(pred_point)
            if a != -1:
                teacher_ask.append(a)
                agent_ask.append(t['agent_ask'][i])
                if agent_ask[-1] == request_index:
                    num_queries += 1
                    if key in ask_action_dict:
                        ask_repeats += request_index in ask_action_dict[key]
                    ask_action_dict[key].add(request_index)
                if hparams.ask_baseline is None:
                    for reason in AskTeacher.reason_labels:
                        agent_reasons[reason].append(reason in t['agent_reason'][i])
                        teacher_reasons[reason].append(reason in t['teacher_reason'][i])
        queries_per_ep.append(num_queries)

    assert sum(queries_per_ep) == sum(agent_ask)

    total_ask = sum(agent_ask) + 1e-8

    loss_str += '\n --- ask:'
    loss_str += ' queries_per_ep %.1f' % (np.average(queries_per_ep))
    loss_str += ', repeat_nav %.2f'    % (nav_repeats / total_nav * 100)
    loss_str += ', repeat_ask %.2f'    % (ask_repeats / total_ask * 100)
    loss_str += ', agent_ratio %.2f'   % (np.average(agent_ask) * 100)
    loss_str += ', teacher_ratio %.2f' % (np.average(teacher_ask) * 100)
    loss_str += ', A/P/R %.2f/%.2f/%.2f' % (
            accuracy_score(teacher_ask, agent_ask),
            precision_score(teacher_ask, agent_ask),
            recall_score(teacher_ask, agent_ask)
        )

    loss_str += '\n --- ask reasons:'

    loss_str += ' ask %.2f, dont_ask %.2f' % (
        np.average(agent_ask) * 100, 100 - np.average(agent_ask) * 100)

    if hparams.ask_baseline is None:
        for reason in AskTeacher.reason_labels:
            loss_str += ', %s %.2f %.2f %.2f/%.2f/%.2f' % (
                reason,
                sum(teacher_reasons[reason]) / len(teacher_reasons[reason]) * 100,
                sum(agent_reasons[reason]) / len(agent_reasons[reason]) * 100,
                accuracy_score(teacher_reasons[reason], agent_reasons[reason]) * 100,
                precision_score(teacher_reasons[reason], agent_reasons[reason]) * 100,
                recall_score(teacher_reasons[reason], agent_reasons[reason]) * 100
            )

    return loss_str

def train(train_env, val_envs, agent, model, optimizer, start_iter, end_iter,
    best_metrics, eval_mode):

    if not eval_mode:
        print('Training with with lr = %f' % optimizer.param_groups[0]['lr'])

    train_feedback = { 'nav': hparams.nav_feedback, 'ask': hparams.ask_feedback }
    test_feedback  = { 'nav': 'argmax', 'ask': 'argmax' }

    start = time.time()
    main_metric = 'success_rate'

    for idx in range(start_iter, end_iter, hparams.log_every):
        interval = min(hparams.log_every, end_iter - idx)

        if eval_mode:
            loss_str = '\n * Eval mode'
        else:
            # Train "interval" iterations
            traj = agent.train(
                train_env, optimizer, idx, idx + interval - 1, train_feedback)

            train_losses = np.array(agent.losses)
            assert len(train_losses) == interval
            train_loss_avg = np.average(train_losses)

            loss_str = '\n * train loss: %.4f' % train_loss_avg
            loss_str += compute_ask_stats(agent, traj)

        metrics = defaultdict(dict)
        should_save_ckpt = []

        for env_name, (env, evaluator) in val_envs.items():
            print('EVAL ' + env_name + '...')

            loss_str += '\n * %s' % env_name.upper()

            # Evaluation
            with torch.no_grad():
                traj = agent.test(env_name, env, test_feedback, idx + interval - 1)

            agent.results_path = os.path.join(hparams.exp_dir,
                '%s_%s_for_eval.json' % (hparams.exp_name, env_name))
            agent.write_results(traj)

            # Compute metrics
            score_summary, _, is_success = evaluator.score(agent.results_path)
            agent.add_is_success(is_success)
            agent.write_results(traj)

            if eval_mode:
                agent.results_path = hparams.load_path.replace('ckpt', '') + env_name + '.json'
                print('Save result to', agent.results_path)
                agent.write_results(traj)

            for metric, val in score_summary.items():
                if metric in ['success_rate', 'oracle_success_rate', 'spl',
                    'oracle_spl', 'target_success_rate', 'error', 'length',
                    'steps']:
                    metrics[metric][env_name] = (val, len(traj))
                if metric in ['success_rate', 'oracle_success_rate', 'spl',
                    'oracle_spl', 'target_success_rate']:
                    loss_str += ', %s: %.2f' % (metric, val * 100)

            # Add info to log string
            loss_str += '\n --- OTHER METRICS: '
            loss_str += '%s: %.2f' % ('error', score_summary['error'])
            loss_str += ', %s: %.2f' % ('oracle_error', score_summary['oracle_error'])
            loss_str += ', %s: %.2f' % ('length', score_summary['length'])
            loss_str += ', %s: %.2f' % ('steps', score_summary['steps'])
            if not (hparams.random_agent or hparams.forward_agent or hparams.shortest_agent):
                loss_str += compute_ask_stats(agent, traj)

            main_metric_value = metrics[main_metric][env_name][0]
            # Add best models to save list
            if not eval_mode and main_metric_value > best_metrics[env_name]:
                should_save_ckpt.append(env_name)
                best_metrics[env_name] = main_metric_value
                print('best %s %s %.2f' %
                    (env_name, main_metric, best_metrics[env_name] * 100))

        if not eval_mode:
            combined_metric = [0, 0]
            for value in metrics[main_metric].values():
                combined_metric[0] += value[0] * value[1]
                combined_metric[1] += value[1]
            combined_metric = combined_metric[0] / combined_metric[1]
            if combined_metric > best_metrics['combined']:
                should_save_ckpt.append('combined')
                best_metrics['combined'] = combined_metric
                print('best combined %s %.2f' % (main_metric, combined_metric * 100))

        iter = idx + interval
        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/end_iter),
            iter, float(iter)/end_iter*100, loss_str))

        if eval_mode:
            res = defaultdict(dict)
            for metric in metrics:
                for k, v in metrics[metric].items():
                    res[metric][k] = v[0]
            return res

        if not eval_mode:
            # Learning rate decay
            if hparams.lr_decay_rate and combined_metric < best_metrics['combined'] \
                and iter >= hparams.start_lr_decay and iter % hparams.decay_lr_every == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= hparams.lr_decay_rate
                    print('New learning rate %f' % param_group['lr'])

            should_save_ckpt.append('last')

            # Save models
            for env_name in should_save_ckpt:
                save_path = os.path.join(hparams.exp_dir,
                    '%s_%s.ckpt' % (hparams.exp_name, env_name))
                save(save_path, model, optimizer, iter, best_metrics, train_env)
                print("Saved %s model to %s" % (env_name, save_path))

        print('\n\n')

    return None


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''

    # Set which GPU to use
    device = torch.device('cuda', hparams.device_id)

    # Load hyperparameters from checkpoint (if exists)
    if os.path.exists(hparams.load_path):
        print('Load model from %s' % hparams.load_path)
        ckpt = load(hparams.load_path, device)
        start_iter = ckpt['iter']
    else:
        if not hparams.forward_agent and not hparams.random_agent and not hparams.shortest_agent:
            if hasattr(hparams, 'load_path') and hasattr(hparams, 'eval_only') and hparams.eval_only:
                sys.exit('load_path %s does not exist!' % hparams.load_path)
        ckpt = None
    start_iter = 0
    end_iter = hparams.n_iters

    if not hasattr(hparams, 'ask_baseline'):
        hparams.ask_baseline = None
    if not hasattr(hparams, 'instruction_baseline'):
        hparams.instruction_baseline = None

    # Set random seeds
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    # Create or load vocab
    train_vocab_path = os.path.join(hparams.data_path, 'vocab.txt')
    if not os.path.exists(train_vocab_path):
        raise Exception('Vocab file not found at %s' % train_vocab_path)
    vocab = read_vocab([train_vocab_path])
    hparams.instr_padding_idx = vocab.index('<PAD>')

    tokenizer = Tokenizer(vocab=vocab, encoding_length=hparams.max_instr_len)
    featurizer = ImageFeatures(hparams.img_features, device)
    simulator = Simulator(hparams)

    # Create train environment
    train_env = Batch(hparams, simulator, featurizer, tokenizer, split='train')

    # Create validation environments
    val_splits = ['val_seen', 'val_unseen']
    eval_mode = hasattr(hparams, 'eval_only') and hparams.eval_only
    if eval_mode:
        if 'val_seen' in hparams.load_path:
            val_splits = ['test_seen']
        elif 'val_unseen' in hparams.load_path:
            val_splits = ['test_unseen']
        else:
            val_splits = ['test_seen', 'test_unseen']
        end_iter = start_iter + 1

    if hparams.eval_on_val:
        val_splits = [x.replace('test_', 'val_') for x in val_splits]

    val_envs_tmp = { split: (
        Batch(hparams, simulator, featurizer, tokenizer, split=split),
        Evaluation(hparams, [split], hparams.data_path))
            for split in val_splits }

    val_envs = {}
    for key, value in val_envs_tmp.items():
        if '_seen' in key:
            val_envs[key + '_env_seen_anna'] = value
            val_envs[key + '_env_unseen_anna'] = value
        else:
            assert '_unseen' in key
            val_envs[key] = value

    # Build model and optimizer
    model = AgentModel(len(vocab), hparams, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.lr,
        weight_decay=hparams.weight_decay)

    best_metrics = { env_name  : -1 for env_name in val_envs.keys() }
    best_metrics['combined'] = -1

    # Load model paramters from checkpoint (if exists)
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        best_metrics = ckpt['best_metrics']
        train_env.ix = ckpt['data_idx']

    if hparams.log_every == -1:
        hparams.log_every = round(len(train_env.data) / \
            (hparams.batch_size * 100)) * 100

    print('')
    pprint(vars(hparams), width=1)
    print('')
    print(model)
    print('Number of parameters:',
        sum(p.numel() for p in model.parameters() if p.requires_grad))

    if hparams.random_agent or hparams.forward_agent or hparams.shortest_agent:
        assert eval_mode
        agent = SimpleAgent(hparams)
    else:
        agent = VerbalAskAgent(model, hparams, device)

    return train(train_env, val_envs, agent, model, optimizer, start_iter,
        end_iter, best_metrics, eval_mode)

if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()

    # Read configuration from a json file
    with open(args.config_file) as f:
        hparams = Namespace(**json.load(f))

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()

    with torch.cuda.device(hparams.device_id):
        train_val()

