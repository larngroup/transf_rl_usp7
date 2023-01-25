# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:43:12 2021

@author: tiago
"""
"""Abstract base model"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, FLAGS):

        @abstractmethod
        def pre_process_data(self):
            pass
    
        @abstractmethod
        def grid_search_cv(self):
            pass
    
        @abstractmethod
        def train_step(self):
            pass