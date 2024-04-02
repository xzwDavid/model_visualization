import streamlit as st
import numpy as np
import pandas as pd
import time

# 假设的模型训练和推理函数
def train_model(epochs):
    for epoch in range(epochs):
        # 模拟训练过程
        loss = np.exp(-epoch / 3.0)  # 假设的损失值
        acc = 1 - loss  # 假设的准确率
        yield epoch, loss, acc

def model_predict(input_data):
    return np.sin(input_data)  # 假设的推理结果


# 假设的模型训练函数，返回每个epoch的train loss和val loss
def train_model(epochs):
    for epoch in range(epochs):
        # 模拟训练过程中的loss变化
        train_loss = np.exp(-epoch / 3.0)  # 假设的train loss
        val_loss = train_loss + 0.1 * np.random.rand()  # 假设的val loss，加入一些随机性
        yield epoch, train_loss, val_loss

# Streamlit 应用
st.title('模型训练监视')

# 模型训练监视
epochs = st.number_input('设置训练的epoch数量', min_value=1, value=10)
if st.button('开始训练模型'):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 准备空的DataFrame来存储loss数据
    data = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Val Loss'])

    chart = st.line_chart(data.set_index('Epoch'))

    for epoch, train_loss, val_loss in train_model(epochs):
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Epoch {epoch + 1}: train loss = {train_loss}, val loss = {val_loss}')

        # 更新数据和图表
        new_data = pd.DataFrame({'Epoch': [epoch + 1], 'Train Loss': [train_loss], 'Val Loss': [val_loss]})
        data = pd.concat([data, new_data])
        chart.line_chart(data.set_index('Epoch'))

        time.sleep(2)  # 模拟训练时间

    status_text.text('训练完成！')


# 模型推理
st.subheader('模型推理')
input_data = st.number_input('输入数据', value=0.0)
if st.button('推理'):
    result = model_predict(input_data)
    st.write(f'推理结果：{result}')

