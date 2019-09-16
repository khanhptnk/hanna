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
from oracle import make_oracle, AskTeacher
from ask_agent import AskAgent


class VerbalAskAgent(AskAgent):

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
            'time_on_task': [0],
            'time': [0],
            'teacher_nav': [],
            'teacher_ask': [],
            'teacher_reason': [],
            'agent_reason': [],
            'agent_reason_prob': [],
            'adj_loc_list': []
        } for ob in obs]

        # Initial decoder states.
        nav_a, ask_a = self.model.reset(self.batch_size)

        ended = [False] * self.batch_size

        should_encode_instruction = True

        info_list = [[] for _ in range(self.batch_size)]
        nav_logits, ask_logits, ask_reason_logits = [], [], []
        nav_pos_targets = []
        last_ask = [-1] * self.batch_size

        device = nav_a.device
        self.nav_loss = torch.tensor(0., device=device)
        self.ask_loss = torch.tensor(0., device=device)
        self.ask_reason_loss = torch.tensor(0., device=device)

        for time_step in range(self.episode_len):

            # Encode instruction
            if should_encode_instruction:
                ctx_seq, ctx_mask = self._text_context_variable(obs)
                nav_ctx, ask_ctx = self.model.encode(ctx_seq, ctx_mask)
                if not self.hparams.no_reset_inter:
                    self.model.reset_text_decoder(self.batch_size)

            # Masks
            nav_a_embeds, nav_logit_mask = self._nav_action_variable(obs)
            ask_logit_mask = self._ask_action_variable(obs)

            # Visual features
            curr_view_features, goal_view_features = \
                self._visual_feature_variable(obs)

            # Time feature
            time = self.from_numpy(np.array([ob['time'] for ob in obs]))
            time_on_task = self.from_numpy(
                np.array([ob['time_on_task'] for ob in obs]))

            # Query nav policy
            nav_logit = self.model.decode_nav(
                time, time_on_task, nav_a, nav_a_embeds, nav_ctx, ctx_mask,
                curr_view_features, goal_view_features, nav_logit_mask)

            # Query nav teacher
            nav_target_list = self.teacher.next_nav(obs)
            nav_a = self._next_action(nav_logit, self.nav_feedback)
            nav_a_list = nav_a.tolist()

            # Compute distribution
            nav_dist = self._compute_nav_dist(obs, nav_logit)
            nav_dist_list = nav_dist.tolist()

            if self.hparams.ask_baseline is None:
                # Query ask policy
                ask_logit, ask_reason_logit = self.model.decode_ask(
                    time, time_on_task, ask_a, nav_dist, ask_ctx, ctx_mask,
                    curr_view_features, goal_view_features, ask_logit_mask)

                ask_logits.append(ask_logit)
                ask_a = self._next_action(ask_logit, self.ask_feedback)
                ask_a_list = ask_a.tolist()

                ask_reason_logits.append(ask_reason_logit)
                ask_reason_prob_list = torch.sigmoid(ask_reason_logit).tolist()
            else:
                # Query ask teacher
                for i, ob in enumerate(obs):
                    ob['last_ask'] = last_ask[i]
                ask_a_list, ask_reason = self.teacher.next_ask(obs)
                for i in range(self.batch_size):
                    ask_a_list[i] = max(0, ask_a_list[i])
                    if ask_a_list[i] == self.ask_actions.index('request_help'):
                        last_ask[i] = time_step

            should_encode_instruction = False
            anna_messages = [None] * self.batch_size

            for i in range(self.batch_size):

                # Perfect language instruction interpretation
                if self.hparams.perfect_interpretation and obs[i]['mode'] == 'on_route':
                    nav_a_list[i] = max(0, nav_target_list[i])

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

            nav_pos_targets.append(np.array(nav_target_list, dtype=np.int64))
            nav_logits.append(nav_logit)

            nav_logit_list = nav_logit.tolist()
            for i in range(self.batch_size):
                info_list[i].append({
                    'ob'        : obs[i],
                    'nav_dist'  : nav_dist_list[i],
                    'nav_target': nav_target_list[i],
                    'nav_a'     : nav_a_list[i],
                    'nav_argmax': int(np.argmax(nav_logit_list[i])),
                    'ask_a'     : ask_a_list[i],
                    'num_a'     : int(nav_logit.size(1))
                })

            # Retrieve embedding of the taken nav action.
            nav_a = nav_a_embeds[np.arange(self.batch_size), nav_a_list, :].detach()

            # Update ask action mask
            ask_a = torch.tensor(ask_a_list, dtype=torch.long, device=device)
            self.model.ask_module.update_action_mask(
                ask_a != self.ask_actions.index('request_help'))

            adj_loc_lists = [ob['adj_loc_list'] for ob in obs]

            # Take the nav action.
            obs = self.env.step(nav_a_list, anna_messages)

            unaligned_nav_dist = F.softmax(nav_logit, dim=1).tolist()

            # Book-keeping
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_pose'].append((
                        ob['viewpoint'], ob['heading'], ob['elevation'], i))
                    traj[i]['agent_mode'].append(ob['mode'])
                    traj[i]['instruction'].append(ob['instruction'])
                    traj[i]['target_viewpoints'].append(ob['target_viewpoints'])
                    traj[i]['message'].append(anna_messages[i])
                    traj[i]['time_on_task'].append(ob['time_on_task'])
                    traj[i]['time'].append(ob['time'])
                    traj[i]['adj_loc_list'].append(adj_loc_lists[i])

                    traj[i]['teacher_nav'].append(nav_target_list[i])

                    if self.hparams.ask_baseline is None:
                        agent_reasons = []
                        out_str = []
                        for k, prob in enumerate(ask_reason_prob_list[i]):
                            label = AskTeacher.reason_labels[k]
                            out_str.append('%s %.1f' % (label[0], prob * 100))
                            if prob >= 0.5:
                                agent_reasons.append(label)
                        traj[i]['agent_reason'].append(agent_reasons)
                        out_str = ' '.join(out_str)
                        traj[i]['agent_reason_prob'].append(out_str)
                    else:
                        traj[i]['teacher_ask'].append(ask_a_list[i])
                        traj[i]['teacher_reason'].append(ask_reason[i])

                    traj[i]['agent_ask'].append(ask_a_list[i])

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

        for i in range(self.batch_size):
            info_list[i].append({ 'ob' : obs[i] })

        # RETROSPECTIVE navigation teacher
        # Look back at the trajectory and decide when the agent should have requested
        if self.hparams.ask_baseline is None:
            ask_targets, ask_reason_targets, ask_reasons = \
                self.teacher.all_ask(info_list)
            for t, target, reason in zip(traj, ask_targets, ask_reasons):
                l = len(t['agent_ask'])
                t['teacher_ask'] = target[:l].tolist()
                t['teacher_reason'] = reason[:l]

        nav_neg_targets, neg_offsets = \
            self.teacher.all_neg_nav(info_list)

        if not self.is_eval:
            # Help-request loss
            if self.hparams.ask_baseline is None:
                # seq_len x batch
                ask_targets = self.from_numpy(ask_targets.transpose())
                ask_reason_targets = self.from_numpy(
                    ask_reason_targets.swapaxes(0, 1)).float()

                for ask_logit, ask_target, ask_reason_logit, ask_reason_target \
                    in zip(ask_logits, ask_targets, ask_reason_logits,
                    ask_reason_targets):

                    # Ask loss
                    ask_loss = self.ask_criterion(ask_logit, ask_target)

                    # Ask reason loss
                    ask_reason_loss = self.ask_reason_criterion(
                        ask_reason_logit, ask_reason_target)
                    ask_reason_loss = ask_reason_loss.mean(dim=-1)
                    mask = (ask_target != -1)
                    normalizer = mask.sum().item()
                    if normalizer > 0:
                        ask_reason_loss = (ask_reason_loss * mask.float()).sum() / \
                            normalizer
                    else:
                        ask_reason_loss = 0.

                    if self.hparams.no_reason:
                        self.ask_loss += ask_loss
                    else:
                        self.ask_loss += ask_loss + ask_reason_loss

            # Navigation loss
            nav_pos_targets = self.from_numpy(np.stack(nav_pos_targets))
            neg_offsets = self.from_numpy(neg_offsets)

            for nav_logit, nav_pos_target, nav_neg_target, neg_offset in \
                zip(nav_logits, nav_pos_targets, nav_neg_targets, neg_offsets):

                # -log P(a+)
                nav_pos_loss = self.nav_criterion(nav_logit, nav_pos_target)

                # K = number of negative actions
                # 1/K sum -log P(a-)
                nav_log_softmax = F.log_softmax(nav_logit, dim=1).view(-1, 1)
                nav_neg_target = self.from_numpy(nav_neg_target)
                nav_neg_loss = -F.embedding_bag(
                    nav_neg_target, nav_log_softmax, neg_offset).squeeze(1)

                mask = (nav_pos_target != -1)
                normalizer = mask.sum().item()
                if normalizer > 0:
                    nav_neg_loss = (nav_neg_loss * mask.float()).sum() / normalizer
                else:
                    nav_neg_loss = 0.

                # nav_loss = -log P(a+) + alpha/K sum log P(a-)
                self.nav_loss += nav_pos_loss - self.hparams.alpha * nav_neg_loss

            self._compute_loss()

        return traj











