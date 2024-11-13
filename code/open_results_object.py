import os
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Importing from the src directory
from src.MOLP import MOLP, Solution
from src.NBI import NBI, plot_NBI_2D, plot_NBI_3D, plot_NBI_3D_to_2D
from src.nNBI import nNBI

# Set up the directories
project_dir = os.getcwd()
code_dir = os.path.join(project_dir, 'code')
data_dir = os.path.join(project_dir, 'data')
data_dir = os.path.join(data_dir, 'melissa_thesis')

with open('melissa_results.pkl', 'rb') as inp:
    nnbi = pickle.load(inp)

plot_NBI_3D(nnbi, normalize_scale=True)