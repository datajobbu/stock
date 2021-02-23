import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


model = keras.models.load_model('lstm/')

start = datetime.date.today() - datetime.timedelta(days=30)
end = datetime.date.today()

df = pdr.DataReader('005930.ks', 'yahoo', start, end)
high = df['High'].values

seq_len = 7

result = []
for idx in range(len(high) - seq_len + 1):
    result.append(high[idx:idx + seq_len])

normalized = []
window_mean = []
window_std = []
for window in result:
    normalized_window = [((p-np.mean(window))/np.std(window)) for p \
                            in window]
    normalized.append(normalized_window)
    window_mean.append(np.mean(window))
    window_std.append(np.std(window))

result = np.array(normalized)

x = np.reshape(result, (result.shape[0], result.shape[1], 1))
pred = model.predict(x)


real = []
pred_result = []
for i in range(len(pred)):
    n1 = (pred[i]*window_std[i]) + window_mean[i]
    n2 = (x[i]*window_std[i]) + window_mean[i]
    pred_result.append(n1)
    real.append(n2)

print(pred_result[-1])
'''fig = plt.figure(figsize=(20,10))
plt.plot(high[-13:], label='True')
plt.plot(pred_result, label='Predict')
plt.legend()
plt.show()'''