import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score
from keras.models import load_model

# load dataset 数据为每个心电图的每个时间点的值，每一行为一张，每一列为一个时间点
train = pd.read_csv('data/ECG200/ECG200_TRAIN.CSV')
test = pd.read_csv('data/ECG200/ECG200_TEST.CSV')

# get target 获取数据集的标签和值
train_target = train.values[:, 0]
test_target = test.values[:, 0]

train_target = (train_target + 1) / 2
test_target = (test_target + 1) / 2

train_value = train.values[:, 1:]
test_value = test.values[:, 1:]
train_value = np.expand_dims(train_value, axis=-1)
test_value = np.expand_dims(test_value, axis=-1)

# 建立时间序列模型
model = Sequential()
model.add(LSTM(256, input_shape=(96, 1)))
model.add(Dense(2, activation='softmax'))

# 训练模型
adam = Adam(lr=0.001)
chk = ModelCheckpoint('model/ecg_model.hdf5', monitor='accuracy', save_best_only=True, mode='max', verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(train_value, train_target, epochs=200, batch_size=80, callbacks=[chk],
          validation_data=(test_value, test_target))

# 测试模型
# loading the model and checking accuracy on the test.py data
model = load_model('model/ecg_model.hdf5')

predictions = model.predict(test_value)
test_label = np.array([int(np.argmax(i)) for i in predictions])
acc_score = accuracy_score(test_target, test_label)
print(acc_score)
