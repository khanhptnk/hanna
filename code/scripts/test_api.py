import sys

import math
sys.path.append('build')

import MatterSim
sim = MatterSim.Simulator()

sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setBatchSize(1)
sim.setCameraResolution(640, 480)
sim.setCameraVFOV(math.radians(60))
sim.setNavGraphPath('../data/connectivity')
sim.initialize()

sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [math.pi/6])

state = sim.getState()

print(state[0].heading, state[0].elevation)

sim.makeAction([0], [-2], [-2])

state = sim.getState()

print(state[0].heading, state[0].elevation)
print(state[0].navigableLocations[0].viewpointId)

from termcolor import colored

print(colored('OKAY', 'green'))
