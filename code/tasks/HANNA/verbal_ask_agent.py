''' Agents: stop/random/shortest/seq2seq  '''

from __future__ import division

import json
import os
import sys
import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from agent import BaseAgent
from oracle import make_oracle
from ask_agent import AskAgent


class VerbalAskAgent(AskAgent):

    def __init__(self, model, hparams, device):

        super(VerbalAskAgent, self).__init__(model, hparams, device)

    def rollout(self):
        # Reset environment.
        obs = self.env.reset()
        self.batch_size = len(obs)

        # Trajectory history.
        traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'agent_pose': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'agent_mode': ['main'],
            'agent_ask': [],
            'agent_nav' : [],
            'instruction': [ob['instruction']],
            'target_viewpoints': [ob['target_viewpoints']],
            'nav_prob': [],
            'message': [],
            'agent_reason': [],
            'teacher_nav': [],
            'adj_loc_list': []
        } for ob in obs]

        # Initial decoder states.
        nav_h, ask_h = self.model.init_state(self.batch_size)

        # Initial actions.
        nav_a, ask_a = self.model.init_action(self.batch_size)

        ended = [False] * self.batch_size

        should_encode_instruction = True

        ask_info_list = [[] for _ in range(self.batch_size)]
        ask_logits = []

        self.nav_loss = self.ask_loss = 0.

        for time_step in range(self.episode_len):

            # Encode instruction
            if should_encode_instruction:
                ctx_seq, ctx_mask = self._text_context_variable(obs)
                nav_ctx, ask_ctx = self.model.encode(ctx_seq)

            # Masks
            nav_a_embeds, nav_logit_mask = self._nav_action_variable(obs)
            ask_logit_mask = self._ask_action_variable(obs)

            # Visual features
            curr_view_features, goal_view_features = \
                self._visual_feature_variable(obs)

            # Query nav policy
            nav_h, nav_logit = self.model.decode_nav(
                nav_h, nav_a, nav_a_embeds, nav_ctx, ctx_mask,
                curr_view_features, goal_view_features, nav_logit_mask)

            # Query nav teacher
            nav_target_list = self.teacher.next_nav(obs)
            nav_a = self._next_action(nav_logit, self.nav_feedback)
            nav_a_list = nav_a.tolist()

            # Compute distribution
            nav_dist = self._compute_nav_dist(obs, nav_logit)
            nav_dist_list = nav_dist.tolist()

            # Query ask policy
            ask_h, ask_logit = self.model.decode_ask(
                ask_h, ask_a, nav_dist, ask_ctx, ctx_mask,
                curr_view_features, goal_view_features, ask_logit_mask)

            ask_logits.append(ask_logit)
            ask_a = self._next_action(ask_logit, self.ask_feedback)
            ask_a_list = ask_a.tolist()

            nav_logit_list = nav_logit.tolist()

            should_encode_instruction = False
            anna_messages = [None] * self.batch_size

            for i in range(self.batch_size):
                # If request
                if ask_a_list[i] == self.ask_actions.index('request_help'):
                    # Query ANNA for route instruction and departure node
                    anna_messages[i] = self.anna(obs[i])
                    # Agent should not move
                    nav_a_list[i] = 0
                    # Teacher nav action should be ignored
                    nav_target_list[i] = -1
                    # Need to re-encode the instruction.
                    should_encode_instruction = True
                else:
                    # If agent decides to depart route, re-encode instruction
                    if nav_a_list[i] == 0 and obs[i]['mode'] == 'on_route':
                        should_encode_instruction = True

            nav_target = torch.tensor(nav_target_list, dtype=torch.long,
                device=self.device)

            # Compute loss.
            if not self.is_eval:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)

            for i in range(self.batch_size):
                ask_info_list[i].append({
                    'ob': obs[i],
                    'nav_dist': nav_dist_list[i],
                    'nav_target': nav_target_list[i],
                    'nav_argmax': int(np.argmax(nav_logit_list[i])),
                    'nav_a': nav_a_list[i],
                    'ask_a': ask_a_list[i]
                })


            # Retrieve embedding of the taken nav action.
            nav_a = nav_a_embeds[np.arange(self.batch_size), nav_a_list, :].detach()

            adj_loc_lists = [ob['adj_loc_list'] for ob in obs]

            # Take the nav action.
            obs = self.env.step(nav_a_list, anna_messages)

            unaligned_nav_dist = F.softmax(nav_logit, dim=1).tolist()

            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_pose'].append((
                        ob['viewpoint'], ob['heading'], ob['elevation'], i))
                    traj[i]['agent_mode'].append(ob['mode'])
                    traj[i]['instruction'].append(ob['instruction'])
                    traj[i]['target_viewpoints'].append(ob['target_viewpoints'])
                    traj[i]['message'].append(anna_messages[i])
                    traj[i]['adj_loc_list'].append(adj_loc_lists[i])

                    traj[i]['teacher_nav'].append(nav_target_list[i])

                    traj[i]['agent_ask'].append(ask_a_list[i])
                    traj[i]['agent_reason'].append([])

                    prob_str = ' '.join(
                        [('%d-%.2f' % (loc['absViewIndex'], x)) for loc, x in
                            zip(adj_loc_lists[i], unaligned_nav_dist[i])])
                    if ask_a_list[i] == self.ask_actions.index('request_help'):
                        traj[i]['agent_nav'].append(-1)
                        traj[i]['nav_prob'].append(prob_str)
                    else:
                        traj[i]['agent_nav'].append(nav_a_list[i])
                        traj[i]['nav_prob'].append('%d %.2f %s' %
                            (nav_a_list[i],
                             unaligned_nav_dist[i][nav_a_list[i]],
                             prob_str))

                    ended[i] |= ob['ended']

            if all(ended):
                break

        # Look back the trajectory and decides when the agent should have asked
        ask_targets, _, ask_reasons = self.teacher.all_ask(ask_info_list)

        for t, target, reason in zip(traj, ask_targets, ask_reasons):
            l = len(t['agent_ask'])
            t['teacher_ask'] = target[:l].tolist()
            t['teacher_reason'] = reason[:l]

        if not self.is_eval:
            # (seq_len x batch) x n_ask_actions
            ask_logits = torch.stack(ask_logits)
            # batch x seq_len
            ask_targets = torch.tensor(ask_targets, dtype=torch.long,
                device=self.device).transpose(0, 1).contiguous()

            for ask_logit, ask_target in zip(ask_logits, ask_targets):
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)

            self._compute_loss()

        return traj












