import numpy as np
from collections import deque

class Queue():

    __init__(self, inputs):
        self.queue = deque(inputs)

    def enqueue(self, a):
        self.queue.append(a)

    def pop(self):
        return self.queue.popleft()
