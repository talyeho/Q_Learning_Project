from dataclasses import dataclass
import numpy as np

# actions according to research book (0, 1, 2, 3, 4, 5)
# actions = ['up', 'down', 'right', 'left', 'forward', 'backward']
N = 4


@dataclass
class State:
    x: float
    y: float
    z: float = 0.0
    actions = np.zeros(6)
    if x == 0:
        actions[3] = -100.0
    if x == N*2:
        actions[2] = -100.0
    if y == 0:
        actions[1] = -100.0
    if y == N*2:
        actions[0] = -100.0
    if z == 0:
        actions[5] = -100.0
    if z == N:
        actions[4] = -100.0
