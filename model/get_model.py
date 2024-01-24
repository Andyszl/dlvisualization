import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import time
import numpy as np
warnings.filterwarnings('ignore')

class DNN_RE_Model(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(DNN_RE_Model, self).__init__()
        layers = []
        prev_dim = input_dim
        for neurons in hidden_layers:
            layers.append(nn.Linear(prev_dim, neurons))
            layers.append(nn.ReLU())
            prev_dim = neurons
        layers.append(nn.Linear(prev_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)