"""Shuffle state advances on optimizer completion, never on worker prefetch."""

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler):
    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.loader_generator = torch.Generator().manual_seed(seed + 1)
        self.epoch = 0
        self.cursor = 0
        self.order = torch.randperm(size, generator=self.generator)

    def __iter__(self):
        for index in self.order[self.cursor:].tolist():
            yield [index]

    def __len__(self):
        return self.size - self.cursor

    def advance(self):
        self.cursor += 1

    def next_epoch(self):
        self.epoch += 1
        self.cursor = 0
        self.order = torch.randperm(self.size, generator=self.generator)

    def state_dict(self):
        return dict(epoch=self.epoch, cursor=self.cursor, order=self.order,
                    generator=self.generator.get_state(), loader_generator=self.loader_generator.get_state())

    def load_state_dict(self, state):
        self.epoch, self.cursor, self.order = state['epoch'], state['cursor'], state['order']
        self.generator.set_state(state['generator'])
        self.loader_generator.set_state(state['loader_generator'])
