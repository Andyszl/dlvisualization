import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import time
import numpy as np
warnings.filterwarnings('ignore')
from basic.data_process import load_data,files_read_excel
from model.get_model import DNN_RE_Model

# Streamlit app
def dnn_app(data,x_col,y_col):
    # Set page title
    st.title('DNN模型训练')

    # Add sidebar for user input
    st.sidebar.title('参数设置')

    # Train ratio slider
    train_ratio = st.sidebar.slider('训练集比例', 0.1, 0.9, 0.8)

    # Batch size slider
    batch_size = st.sidebar.slider('批量大小', 1, 100, 32)

    # Learning rate slider
    learning_rate = st.sidebar.slider('学习率', 0.001, 0.1, 0.01)

    # Epochs slider
    epochs = st.sidebar.slider('时期数', 10, 500, 100)

    # Hidden layers and neurons input
    num_hidden_layers = st.sidebar.slider('隐藏层层数', 1, 5, 2)
    hidden_layers = []
    for i in range(num_hidden_layers):
        neurons = st.sidebar.slider(f'隐藏层 {i+1} 神经元个数', 5, 100, 10, key=i)
        hidden_layers.append(neurons)

    # Load data based on user input
    train_ds,train_dl,test_ds,test_dl= load_data(data,x_col,y_col,batch_size, train_ratio)

    # Instantiate the model
    model = DNN_RE_Model(len(x_col), hidden_layers)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    create_button = st.button("开始训练")

    # 创建网络或获取已保存的网络
    if create_button:
        # Train the model
        progress_bar = st.progress(0)
        status_text = st.empty()
        # 生成示例数据
        data_plot = {'Index': [],
                'train_loss': []
                ,'test_loss' : []}
        df_plot = pd.DataFrame(data_plot)

        # 创建折线图
        chart = st.line_chart(df_plot)

        for epoch in range(epochs):
            for xb, yb in train_dl:

                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                train_epoch_loss = sum(criterion(model(xb), yb) for xb, yb in train_dl)
                test_epoch_loss = sum(criterion(model(xb), yb) for xb, yb in test_dl)


            new_data = {'Index': epoch,
                        'train_loss': train_epoch_loss.data.item() / len(train_dl)
                        ,'test_loss': test_epoch_loss.data.item() / len(test_dl)}
            new_row = pd.DataFrame([new_data])
            status_text.text("%i%% Complete" % ((epoch+1)* 100/epochs))
            progress_bar.progress(round((epoch+1)* 100/epochs))
            # 添加新行到折线图
            chart.add_rows(new_row[["train_loss","test_loss"]])
            #time.sleep(0.1)
        progress_bar.empty()


# 配置页面
st.set_page_config(
    page_title="DNN 回归",
    layout="wide",
    initial_sidebar_state="expanded",
)


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


st.header('一、 建议&问题清单抽取', divider='rainbow')

st.subheader('1 数据上传', divider='gray')
# 1、上传 文件
excel_file = st.file_uploader('上传CSV文件:', type='csv', key = "up_one")
if excel_file is not None:
    data = files_read_excel(excel_file)
    st.write("数据量: ", len(data))

# 2、模型设置
st.subheader('1.3 模型训练', divider='gray')

## 2.1 选择数据
if excel_file is not None:

    col_options = [""] + list(data.columns)
    selected_X = st.multiselect('请选择自变量字段:',col_options, key = "seld_two")
    st.write(selected_X)
    options_y = [item for item in col_options if item not in selected_X]
    selected_y = st.selectbox(
        "请选择标签值", options_y, key = "seld_three"
    )
    if not selected_X or selected_X == "" or not selected_y:
        st.error("请先选择数据...")
    else:

        dnn_app(data,selected_X,selected_y)

