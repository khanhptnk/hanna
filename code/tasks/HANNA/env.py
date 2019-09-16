import os
import sys
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
from collections import defaultdict, namedtuple
import scipy.stats
import copy
from functools import partial

sys.path.append('../../build')
import MatterSim
import torch

from oracle import make_oracle
from utils import load_datasets, load_nav_graphs
import utils

csv.field_size_limit(sys.maxsize)

angle_inc = np.pi / 6.
NUM_VIEWS = 36
MEAN_POOLED_DIM = 2048

IMAGE_W = 640
IMAGE_H = 480
VFOV = 60

def _build_action_embedding(adj_loc_list, features, loc_embed_size=None):
    feature_dim = features.shape[-1]
    embedding = np.zeros((len(adj_loc_list), feature_dim + loc_embed_size), np.float32)
    for a, adj_dict in enumerate(adj_loc_list):
        if a == 0:
            # the embedding for the first action ('stop') is left as zero
            continue
        embedding[a, :feature_dim] = features[adj_dict['absViewIndex']]
        loc_embedding = embedding[a, feature_dim:]
        rel_heading = adj_dict['rel_heading']
        rel_elevation = adj_dict['rel_elevation']
        shift = loc_embed_size // 4
        loc_embedding[0          :shift]       = math.sin(rel_heading)
        loc_embedding[shift      :(shift * 2)] = math.cos(rel_heading)
        loc_embedding[(shift * 2):(shift * 3)] = math.sin(rel_elevation)
        loc_embedding[(shift * 3):]            = math.cos(rel_elevation)
    return embedding

def _calculate_headings_and_elevations_for_views(sim, goalViewIndices):
    states = sim.getState()
    heading_deltas = []
    elevation_deltas = []
    for state, goalViewIndex in zip(states, goalViewIndices):
        currViewIndex = state.viewIndex
        heading_deltas.append(goalViewIndex % 12 - currViewIndex % 12)
        elevation_deltas.append(goalViewIndex // 12 - currViewIndex // 12)
    return heading_deltas, elevation_deltas


class ImageFeatures(object):
    '''
       Load and manage visual features (pretrained by ResNet-152 on ImageNet)
    '''

    def __init__(self, image_feature_file, device):
        print('Loading image features from %s' % image_feature_file)
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
        self.device = device

        not_found_features = np.zeros(
            (NUM_VIEWS + 1, MEAN_POOLED_DIM), dtype=np.float32)
        self.features = defaultdict(lambda: not_found_features)

        with open(image_feature_file, "rt") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
            for item in reader:

                assert int(item['image_h']) == IMAGE_H
                assert int(item['image_w']) == IMAGE_W
                assert int(item['vfov']) == VFOV

                long_id = self._make_id(item['scanId'], item['viewpointId'])
                features = np.frombuffer(utils.decode_base64(item['features']),
                    dtype=np.float32).reshape((NUM_VIEWS, MEAN_POOLED_DIM))
                no_look_feature = np.zeros(
                    (1, MEAN_POOLED_DIM), dtype=np.float32)
                features = np.concatenate((features, no_look_feature), axis=0)
                self.features[long_id] = features

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def get_features(self, scanId, viewpointId):
        long_id = self._make_id(scanId, viewpointId)
        return self.features[long_id]


class Simulator():

    def __init__(self, hparams):

        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(IMAGE_W, IMAGE_H)
        self.sim.setCameraVFOV(math.radians(VFOV))
        self.sim.setNavGraphPath(os.path.join(hparams.data_path, '../connectivity'))
        self.sim.setBatchSize(hparams.batch_size)
        self.sim.initialize()

        self.batch_size = hparams.batch_size

        print('Loading adjacent lists for panoramic action spaces')
        with open(os.path.join(hparams.data_path, 'panoramic_action_space.json')) as f:
            self.cached_adj_loc_lists = json.load(f)

    def new_episodes(self, states):
        self.sim.newEpisode(*zip(*states))
        self.states = self.sim.getState()

    def navigate_to_locations(self,
            nextViewpointIds, nextViewIndices, navigableViewIndices):

        # Rotate to the view index assigned to the next viewpoint
        heading_deltas, elevation_deltas = \
            _calculate_headings_and_elevations_for_views(
                self.sim, navigableViewIndices)

        self.sim.makeAction(
            [0] * self.batch_size, heading_deltas, elevation_deltas)

        states = self.sim.getState()
        locationIds = []
        for i, (state, nextViewpointId, navigableViewIndex) in \
            enumerate(zip(states, nextViewpointIds, navigableViewIndices)):

            # Check if rotation was done right
            assert state.viewIndex == navigableViewIndex

            # Find index of the next viewpoint
            index = None
            for i, loc in enumerate(state.navigableLocations):
                if loc.viewpointId == nextViewpointId:
                    index = i
                    break
            assert index is not None
            locationIds.append(index)

        # Rotate to the target view index
        heading_deltas, elevation_deltas = \
            _calculate_headings_and_elevations_for_views(
                self.sim, nextViewIndices)

        self.sim.makeAction(
            locationIds, heading_deltas, elevation_deltas)

        # Final check
        self.states = self.sim.getState()
        for state, nextViewpointId, nextViewIndex in \
            zip(self.states, nextViewpointIds, nextViewIndices):

            assert state.viewIndex == nextViewIndex
            assert state.location.viewpointId == nextViewpointId

    def get_panorama_states(self):
        self.adj_loc_lists = []
        for state in self.states:
            long_id = '_'.join([state.scanId,
                state.location.viewpointId, str(state.viewIndex % 12)])
            self.adj_loc_lists.append(self.cached_adj_loc_lists[long_id])

        return self.states, self.adj_loc_lists


class Batch():

    def __init__(self, hparams, simulator, featurizer, tokenizer, split=None):

        self.sim = simulator

        self.random = random
        self.random.seed(hparams.seed)

        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.split = split
        self.batch_size = hparams.batch_size

        self.build_action_embeds = partial(_build_action_embedding,
            loc_embed_size=hparams.loc_embed_size)

        self.load_data(load_datasets([split], hparams.data_path,
            prefix=hparams.data_prefix))

        self.max_queries = None

    def encode(self, instr):
        assert self.tokenizer is not None, 'No tokenizer'
        return self.tokenizer.encode_sentence(instr)

    def load_data(self, data):
        self.data = []
        for item in data:
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                del new_item['instructions']
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                self.data.append(new_item)

        self.reset_epoch()

        if self.split is not None:
            print('Dataset loaded with %d instructions, using split: %s' % (
                len(self.data), self.split))

    def _next_minibatch(self):
        if self.ix == 0:
            self.random.shuffle(self.data)
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            self.random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        self.ix = 0

    def make_obs(self):
        obs = []

        self.sim.get_panorama_states()
        states, adj_loc_lists = self.sim.states, self.sim.adj_loc_lists
        for i, (state, adj_loc_list) in enumerate(zip(states, adj_loc_lists)):

            item = self.batch[i]

            curr_view_features = self.featurizer.get_features(state.scanId,
                state.location.viewpointId)

            goal_view_features = self.featurizer.get_features(state.scanId,
                self.target_viewpoint_with_features[i])

            action_embeds = self.build_action_embeds(
                adj_loc_list, curr_view_features)

            obs.append({
                'instr_id' : item['instr_id'],
                'subgoal_instr_id': self.subgoal_instr_id[i],
                'scan' : item['scan'],
                'mode' : self.mode[i],
                'time' : self.time[i],
                'time_on_task': self.time_on_task[i],
                'ended': self.ended[i],
                'viewpoint' : state.location.viewpointId,
                'view_index' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'curr_view_features' : curr_view_features,
                'goal_view_features' : goal_view_features,
                'action_embeds': action_embeds,
                'adj_loc_list': adj_loc_list,
                'instruction' : self.instruction[i],
                'init_viewpoint' : item['paths'][0][0],
                'target_viewpoints': self.target_viewpoints[i],
                'goal_viewpoints': self.goal_viewpoints[i],
            })

        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()

        init_states = [[item['scan'], item['paths'][0][0], item['heading'], 0]
            for item in self.batch]
        self.sim.new_episodes(init_states)

        self.mode = ['main'] * self.batch_size
        self.time = [0] * self.batch_size
        self.time_on_task = [0] * self.batch_size
        self.ended = [False] * self.batch_size

        self.instruction = [item['instruction'] for item in self.batch]
        self.anna_message = [None] * self.batch_size
        self.subgoal_instr_id = ['none'] * self.batch_size

        self.target_viewpoint_with_features = ['none'] * self.batch_size
        self.goal_viewpoints = []
        for item in self.batch:
            self.goal_viewpoints.append([p[-1] for p in item['paths']])
        self.target_viewpoints = copy.deepcopy(self.goal_viewpoints)

        return self.make_obs()

    def step(self, nav_actions, anna_messages):

        navigableViewIndices = []
        nextViewpointIds = []
        nextViewIndices = []
        states, adj_loc_lists = self.sim.states, self.sim.adj_loc_lists
        for i in range(self.batch_size):
            # Increment time
            self.time[i] += 1
            message = anna_messages[i]
            if message is None:
                if nav_actions[i] == 0:
                    # Agent wants to depart route
                    if self.mode[i] == 'on_route':
                        self.depart_route(i)
                    else:
                        # Agent wants to end episode
                        assert self.mode[i] == 'main'
                        self.ended[i] = True
                    # Agent stays at the same location
                    nextViewpointIds.append(states[i].location.viewpointId)
                    nextViewIndices.append(states[i].viewIndex)
                    navigableViewIndices.append(states[i].viewIndex)
                else:
                    self.time_on_task[i] += 1
                    # Agent chooses a next location
                    loc = adj_loc_lists[i][nav_actions[i]]
                    nextViewpointIds.append(loc['nextViewpointId'])
                    nextViewIndices.append(loc['absViewIndex'])
                    navigableViewIndices.append(loc['absViewIndex'])
            else:
                # Agent requests help and enters route
                nextViewpointIds.append(message['start_node'])
                nextViewIndices.append(message['view_id'])
                # Find viewIndex that contains start_node
                navigableViewIndex = None
                for loc in adj_loc_lists[i]:
                    if loc['nextViewpointId'] == nextViewpointIds[-1]:
                        navigableViewIndex = loc['absViewIndex']
                        break
                assert navigableViewIndex is not None
                navigableViewIndices.append(navigableViewIndex)
                self.enter_route(i, message)

        # Execute navigation action
        self.sim.navigate_to_locations(
            nextViewpointIds, nextViewIndices, navigableViewIndices)

        return self.make_obs()

    def enter_route(self, i, message):
        self.mode[i] = 'on_route'
        # Reset local time
        self.time_on_task[i] = 0
        # Set subtask
        self.instruction[i] = message['instruction']
        self.target_viewpoints[i] = [message['depart_node']]
        self.target_viewpoint_with_features[i] = self.target_viewpoints[i][0]
        self.subgoal_instr_id[i] = message['path_id']
        self.anna_message[i] = message

    def depart_route(self, i):
        self.mode[i] = 'main'
        # Reset local time
        self.time_on_task[i] = 0
        # Resume main task
        self.instruction[i] = self.batch[i]['instruction']
        self.target_viewpoints[i] = [self.anna_message[i]['goal_node']]
        self.target_viewpoint_with_features[i] = self.target_viewpoints[i][0]
        self.subgoal_instr_id[i] = 'none'
        self.anna_message[i] = None






