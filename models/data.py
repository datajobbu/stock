import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import optimizers


class prediction_Model(Model):
    def __init__(self):
        super(prediction_Model, self).__init__()
        self.lstm = LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True) # return_sequences=False  : single LSTM
        self.lstm2 = LSTM(64,)
        self.layer1 = Dense(32, activation='relu')
        self.layer2 = Dense(16, activation='relu')
        self.layer3 = Dense(1)

    def call(self, inputs):
        lstm = self.lstm(inputs)
        lstm2 = self.lstm2(lstm)
        layer1 = self.layer1(lstm2)
        layer2 = self.layer2(layer1)
        out = self.layer3(layer2)
        return out


start = datetime.date.today() - datetime.timedelta(days=5*365)
end = datetime.date.today()
split = pd.Timestamp('01-01-2021')

df = pdr.DataReader('005930.ks', 'yahoo', start, end)

high = df['High'].values

seq_len = 7
seq_length = seq_len + 1

result = []

for idx in range(len(high) - seq_len):
    result.append(high[idx:idx + seq_length])

normalized = []
window_mean = []
window_std = []
for window in result:
    normalized_window = [((p-np.mean(window))/np.std(window)) for p in window]
    normalized.append(normalized_window)
    window_mean.append(np.mean(window))
    window_std.append(np.std(window))

result = np.array(normalized)
row = int(round(result.shape[0]*0.9))
train = result[:row, :]
test = result[row:, :]
window_mean = window_mean[row:]
window_std = window_std[row:]

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = test[:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = test[:, -1]

model = prediction_Model()
model.compile(loss='mae', optimizer=optimizers.Adam(lr=0.001))
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          batch_size=64, epochs=300)

pred = model.predict(x_test)

pred_result = []
pred_y = []
for i in range(len(pred)):
    n1 = pred[i] * window_std[i] + window_mean[i]
    n2 = y_test[i] * window_std[i] + window_mean[i]

    pred_result.append(n1)
    pred_y.append(n2)

print(pred_result[-1], pred_y[-1])

'''fig = plt.figure(figsize=(20,10))
plt.plot(y_test, label='True')
plt.plot(pred, label='Predict')
plt.legend()
plt.show()'''

model.save('lstm', save_format='tf')