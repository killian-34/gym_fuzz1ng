import numpy as np
from collections import deque

class Queue():

    def __init__(self, inputs):
        self.queue = deque(inputs)

    def enqueue(self, a):
        self.queue.append(a)

    def pop(self):
        return self.queue.popleft()


def deterministic_edits(input):
    yield input
    for i in range(5):#len(input)):
        new_input = np.copy(input)
        new_input[i] = 1
        yield new_input