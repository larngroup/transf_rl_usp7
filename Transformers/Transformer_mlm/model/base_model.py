# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:43:12 2021

@author: tiago
"""
"""Abstract base model"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, cfg):
        #self.config = Config.from_json(cfg)
        self.config = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass