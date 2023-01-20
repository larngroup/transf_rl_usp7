# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:22:07 2021

@author: tiago
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import csv
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
from bunch import Bunch
from tqdm import tqdm

class configs:
    """Configuration loader class"""

    @staticmethod
    def load_cfg(path):
        """ Loads configuration file
        Args:
            path (str): The path of the configuration file

        Returns:
            config (json): The configuration file
        """
        with open(path, 'r') as config_file:
            config_dict = json.load(config_file)
            config = Bunch(config_dict)
        return config