import os
import sys
sys.path.append('build')
import MatterSim
import time
import math
import cv2
import numpy as np

WIDTH = 800
HEIGHT = 600
VFOV = math.radians(60)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]
DISCRETIZE_VIEW = True

cv2.namedWindow('displaywin')
sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setNavGraphPath(os.path.join('../data', 'connectivity'))
sim.setDiscretizedViewingAngles(DISCRETIZE_VIEW)
sim.initialize()

print('\nPython Demo')
print('Use arrow keys to move the camera.')
print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.\n')

if len(sys.argv) > 1:
    house_id = sys.argv[1]
else:
    house_id = '17DRP5sb8fy'

if len(sys.argv) <= 2:
    sim.newRandomEpisode([house_id])
else:
    view_id = sys.argv[2]
    heading = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    sim.newEpisode([house_id], [view_id], [heading], [0])


heading = 0
elevation = 0
location = 0
ANGLEDELTA = 5 * math.pi / 180

vp_id = None
heading_id = heading
elevation_id = elevation

while True:
    sim.makeAction([location], [heading], [elevation])
    location = 0
    heading = 0
    elevation = 0
    state = sim.getState()[0]

    if state.location.viewpointId != vp_id or abs(state.heading - heading_id) > 1e-6 \
       or abs(state.elevation - elevation_id) > 1e-6:
        print(abs(state.heading - heading_id))
        vp_id = state.location.viewpointId
        heading_id = state.heading
        elevation_id = state.elevation
        print(vp_id, heading_id, elevation_id, state.viewIndex, state.location)

    locations = state.navigableLocations
    im = np.array(state.rgb, copy=False)
    origin = (locations[0].x, locations[0].y, locations[0].z)
    for idx, loc in enumerate(locations[1:]):
        # Draw actions on the screen
        fontScale = 3.0/loc.rel_distance
        x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
        y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
        cv2.putText(im, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, TEXT_COLOR, thickness=3)
        cv2.putText(im, str(loc.viewpointId), (x-5, y - 100 + 25 * idx), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, [255, 252, 0], thickness=1)

    cv2.imshow('displaywin', im)
    k = cv2.waitKey(1)
    if k == -1:
        continue
    else:
        k = (k & 255)
    if k == ord('q'):
        break
    elif ord('1') <= k <= ord('9'):
        location = k - ord('0')
        if location >= len(locations):
            location = 0
    elif k == 81 or k == ord('a'):
        heading = -1 if DISCRETIZE_VIEW else -ANGLEDELTA
    elif k == 82 or k == ord('w'):
        elevation = 1 if DISCRETIZE_VIEW else ANGLEDELTA
    elif k == 83 or k == ord('d'):
        heading = 1 if DISCRETIZE_VIEW else ANGLEDELTA
    elif k == 84 or k == ord('s'):
        elevation = -1 if DISCRETIZE_VIEW else -ANGLEDELTA
