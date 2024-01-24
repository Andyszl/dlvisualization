import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import time
import numpy as np
warnings.filterwarnings('ignore')


def files_read_excel(files):
    if files is not None:
        st.write(files.name)
        #读取excel
        csv_reader = pd.read_csv(files)
        text = csv_reader.head(10)
        st.write(text)
    return csv_reader

def load_data(data,x_col,y_col, batch_size, train_ratio=0.8):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader

    train,test = train_test_split(data, train_size=train_ratio)

    train_x = train[x_col].values
    test_x = test[x_col].values

    train_y = train[y_col].values.reshape(-1, 1)
    test_y = test[y_col].values.reshape(-1, 1)

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    test_y = torch.from_numpy(test_y).type(torch.FloatTensor)



    train_ds = TensorDataset(train_x, train_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(test_x, test_y)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)


    return train_ds,train_dl,test_ds,test_dl