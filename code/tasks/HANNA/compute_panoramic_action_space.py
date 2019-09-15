import sys
import json
import os
import math
import numpy as np
from pprint import pprint

sys.path.append('../../build')

import MatterSim


angle_inc = np.pi / 6.

def _canonical_angle(x):
    ''' Make angle in (-pi, +pi) '''
    return x - 2 * math.pi * round(x / (2 * math.pi))

def get_panorama_states(scan, viewpoint, init_view_index):

    init_heading = (init_view_index % 12) * angle_inc
    init_elevation = (init_view_index // 12 - 1) * angle_inc

    sim.newEpisode([scan], [viewpoint], [init_heading], [init_elevation])

    state = sim.getState()[0]

    assert state.viewIndex == init_view_index

    elevation_delta = -(init_view_index // 12)

    assert elevation_delta == 0

    adj_dict = {}

    for relViewIndex in range(36):
        # Here, base_rel_heading and base_rel_elevation are w.r.t
        # relViewIndex 12 (looking forward horizontally)
        # (i.e. the relative heading and elevation
        # adjustment needed to switch from relViewIndex 12
        # to the current relViewIndex)
        base_rel_heading = (relViewIndex % 12) * angle_inc
        base_rel_elevation = (relViewIndex // 12 - 1) * angle_inc

        state = sim.getState()[0]
        absViewIndex = state.viewIndex
        for loc in state.navigableLocations[1:]:
            viewpointId = loc.viewpointId
            rel_heading = loc.rel_heading
            rel_elevation = loc.rel_elevation

            distance = math.sqrt(rel_heading ** 2 + rel_elevation ** 2)

            if viewpointId not in adj_dict or \
                distance < adj_dict[viewpointId]['distance']:

                adj_dict[viewpointId] = {
                    'relViewIndex': relViewIndex,
                    'absViewIndex': absViewIndex,
                    'nextViewpointId': viewpointId,
                    'rel_heading': _canonical_angle(base_rel_heading + rel_heading),
                    'rel_elevation': base_rel_elevation + rel_elevation,
                    'distance': distance
                }

        # Move to the next view
        if (relViewIndex + 1) % 12 == 0:
            sim.makeAction([0], [1], [1])
        else:
            sim.makeAction([0], [1], [0])

    sim.makeAction([0], [0], [-2 - elevation_delta])

    state = sim.getState()[0]

    # Check if the pose has been restored
    assert state.viewIndex == init_view_index

    # collect navigable location list
    stop = {
        'relViewIndex': -1,
        'absViewIndex': state.viewIndex,
        'nextViewpointId': state.location.viewpointId}

    adj_loc_list = [stop] + sorted(
        adj_dict.values(), key=lambda x: abs(x['rel_heading']))

    return adj_loc_list


IMAGE_W = 640
IMAGE_H = 480
VFOV = 60

sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setCameraResolution(IMAGE_W, IMAGE_H)
sim.setCameraVFOV(math.radians(VFOV))
sim.setNavGraphPath(os.path.join(
    os.getenv('PT_DATA_DIR', '../../../data'), 'connectivity'))
sim.setBatchSize(1)
sim.initialize()

DATA_PATH = '../../../data/anna'

with open(os.path.join(DATA_PATH, 'scan_split.json')) as f:
    scan_split = json.load(f)

scans = scan_split['train'] + scan_split['val'] + scan_split['test']


output = {}

for scan in scans:
    print(scan)
    with open(os.path.join(DATA_PATH, '../connectivity/' + scan + '_connectivity.json')) as f:
        graph_data = json.load(f)
        for item in graph_data:
            if not item['included']: continue
            view = item['image_id']
            for view_index in range(12):
                long_id = '_'.join([scan, view, str(view_index)])
                output[long_id] = get_panorama_states(scan, view, view_index)

print(len(output))

with open(os.path.join(DATA_PATH, 'panoramic_action_space.json'), 'w') as f:
    json.dump(output, f, indent=4, sort_keys=True)

