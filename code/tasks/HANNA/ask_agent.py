import json
import os
import sys
import numpy as np
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from agent import BaseAgent
from oracle import make_oracle


class AskAgent(BaseAgent):

    ask_actions = ['do_nothing', 'request_help', '<start>']
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, model, hparams, device):
        super(AskAgent, self).__init__()

        self.model = model

        # Losses
        self.nav_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.ask_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.ask_reason_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Oracles
        self.env_oracle = make_oracle('env_oracle', hparams.scan_path)
        self.anna = make_oracle('anna', hparams, self.env_oracle)
        self.teacher = make_oracle('teacher', hparams, self.ask_actions,
            self.env_oracle, self.anna)

        self.device = device

        self.from_numpy = lambda array: \
            torch.from_numpy(array).to(self.device)

        self.hparams = hparams

    @staticmethod
    def n_output_nav_actions():
        return 37

    @staticmethod
    def n_input_ask_actions():
        return len(AskAgent.ask_actions)

    @staticmethod
    def n_output_ask_actions():
        return len(AskAgent.ask_actions) - 1

    def _text_context_variable(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
             sequence length (to enable PyTorch packing). '''

        def encode_batch(seq_list):
            seq_tensor = np.array(seq_list)
            seq_lengths = np.argmax(seq_tensor == self.hparams.instr_padding_idx, axis=1)
            seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

            max_length = max(seq_lengths)
            assert max_length <= self.hparams.max_instr_len

            seq_tensor  = self.from_numpy(seq_tensor).long()[:,:max_length]
            seq_lengths = self.from_numpy(seq_lengths).long()

            seq_mask = (seq_tensor == self.hparams.instr_padding_idx)

            return seq_tensor, seq_mask

        nav_seq_list = []
        for ob in obs:
            instruction = ob['instruction']
            if self.hparams.instruction_baseline == 'vision_only' and ob['mode'] == 'on_route':
                instruction = ''
            nav_seq_list.append(self.env.encode(instruction))

        return encode_batch(nav_seq_list)

    def _visual_feature_variable(self, obs):

        def to_tensor(feature_tuple):
            return self.from_numpy(np.stack(feature_tuple))

        curr_view_feature_tuple = tuple(ob['curr_view_features'] for ob in obs)
        goal_view_feature_tuple = tuple(ob['goal_view_features'] for ob in obs)

        return to_tensor(curr_view_feature_tuple), \
               to_tensor(goal_view_feature_tuple)

    def _nav_action_variable(self, obs):
        max_num_a = max(len(ob['adj_loc_list']) for ob in obs)
        invalid = np.zeros((self.batch_size, max_num_a), np.uint8)
        action_embed_dim = obs[0]['action_embeds'].shape[-1]
        action_embeds = np.zeros(
            (self.batch_size, max_num_a, action_embed_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            invalid[i, num_a:] = 1
            action_embeds[i, :num_a, :] = ob['action_embeds']
        return self.from_numpy(action_embeds), self.from_numpy(invalid)

    def _ask_action_variable(self, obs):
        ask_logit_mask = torch.zeros(self.batch_size,
            AskAgent.n_output_ask_actions(),
            dtype=torch.uint8, device=self.device)

        ask_mask_indices = []
        for i, ob in enumerate(obs):
            if ob['ended'] or \
                not self.anna.can_request(ob['scan'], ob['viewpoint']):
                ask_mask_indices.append(
                    (i, self.ask_actions.index('request_help')))

        ask_logit_mask[list(zip(*ask_mask_indices))] = 1

        return ask_logit_mask

    def _argmax(self, logit):
        return logit.max(1)[1].detach()

    def _sample(self, logit):
        prob = F.softmax(logit, dim=1).contiguous()

        # Weird bug with torch.multinomial: it samples even zero-prob actions.
        while True:
            sample = torch.multinomial(prob, 1, replacement=True).view(-1)
            is_good = True
            for i in range(logit.size(0)):
                if logit[i, sample[i].item()].item() == -float('inf'):
                    is_good = False
                    break
            if is_good:
                break

        return sample

    def _next_action(self, logit, feedback):
        if feedback == 'argmax':
            return self._argmax(logit)
        if feedback == 'sample':
            return self._sample(logit)
        sys.exit('Invalid feedback option')

    def _compute_nav_dist(self, obs, nav_logit):
        nav_softmax = F.softmax(nav_logit, dim=1).tolist()
        nav_softmax_full = np.zeros((self.batch_size,
            AskAgent.n_output_nav_actions()), dtype=np.float32)
        for i, ob in enumerate(obs):
            for p, adj_dict in zip(nav_softmax[i], ob['adj_loc_list']):
                k = adj_dict['relViewIndex']
                nav_softmax_full[i, k] += p
        return self.from_numpy(nav_softmax_full)

    def _compute_loss(self):
        self.loss = self.nav_loss + self.ask_loss
        self.losses.append(self.loss.item() / self.episode_len)

        self.nav_losses.append(self.nav_loss.item() / self.episode_len)
        self.ask_losses.append(self.ask_loss.item() / self.episode_len)

    def _setup(self, env, feedback):
        self.nav_feedback = feedback['nav']
        self.ask_feedback = feedback['ask']

        assert self.nav_feedback in self.feedback_options
        assert self.ask_feedback in self.feedback_options

        self.env = env
        self.env.env_oracle = self.env_oracle
        self.losses = []
        self.nav_losses = []
        self.ask_losses = []

    def test(self, env_name, env, feedback, iter):
        self.is_eval = True
        self._setup(env, feedback)
        self.model.eval()

        self.episode_len = self.hparams.eval_episode_len

        self.anna.is_eval = True
        self.cached_results = defaultdict(dict)

        if '_seen_anna' in env_name:
            self.anna.split_name = 'train_seen'
        elif '_unseen_anna' in env_name:
            self.anna.split_name = 'train_unseen'
        elif env_name == 'val_unseen':
            self.anna.split_name = 'val'
        elif env_name == 'test_unseen':
            self.anna.split_name = 'test'
        else:
            raise Exception('env_name not found %s' % env_name)

        return BaseAgent.test(self)

    def train(self, env, optimizer, start_iter, end_iter, feedback):
        self.is_eval = False
        self._setup(env, feedback)
        self.model.train()

        self.anna.is_eval = False
        self.anna.split_name = 'train_seen'

        self.episode_len = self.hparams.train_episode_len

        last_traj = []
        for iter in range(start_iter, end_iter + 1):
            optimizer.zero_grad()
            traj = self.rollout()
            if end_iter - iter + 1 <= 10:
                last_traj.extend(traj)
            self.loss.backward()
            optimizer.step()

        return last_traj


class SimpleAgent(BaseAgent):

    def __init__(self, hparams):
        super(SimpleAgent, self).__init__()

        self.random = random
        self.random.seed(hparams.seed)
        self.episode_len = hparams.eval_episode_len

        self.hparams = hparams

        if self.hparams.shortest_agent:
            self.env_oracle = make_oracle('env_oracle', hparams.scan_path)
            self.nav_teacher = make_oracle('nav', self.env_oracle)

    def test(self, env_name, env, feedback, iter):
        self.is_eval = True
        self.env = env
        self.losses = [0]
        self.nav_losses = [0]

        return BaseAgent.test(self)

    def rollout(self):
        # Reset environment.
        obs = self.env.reset()
        batch_size = len(obs)

        # Trajectory history.
        traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'agent_pose': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'agent_nav' : []
        } for ob in obs]

        ended = [False] * batch_size

        for time_step in range(self.episode_len):
            nav_a_list = [None] * batch_size
            if self.hparams.shortest_agent:
                nav_a_list = self.nav_teacher(obs)
            else:
                for i, ob in enumerate(obs):
                    if self.hparams.random_agent:
                        nav_a_list[i] = self.random.randint(
                            0, len(ob['adj_loc_list']) - 1)
                    else:
                        assert self.hparams.forward_agent
                        if len(ob['adj_loc_list']) > 1 and time_step < 10:
                            nav_a_list[i] = 1
                        else:
                            nav_a_list[i] = 0

            obs = self.env.step(nav_a_list, [None] * len(obs))

            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_pose'].append((
                        ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['agent_nav'].append(nav_a_list[i])

                ended[i] |= nav_a_list[i] == 0

            if all(ended):
                break

        return traj


