import torch.nn as nn
import torch
from abc import ABCMeta, abstractmethod


class IEModel(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def generate_batch(batch_data):
        '''
        generate batch data for training
        :return:
        '''
        pass


class TPLinkerPlus(nn.Module, IEModel):
    def __init__(self):
        super().__init__()
        print("init")
        pass

    def forward(self):
        print("forward")
        pass

    @staticmethod
    def generate_batch(batch_data):
        pass


class TriggerFreeEventExtraction(nn.Module, IEModel):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    @staticmethod
    def generate_batch(batch_data):
        pass