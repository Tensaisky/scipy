import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras import layers
from decimal import Decimal
from keras.optimizers import RMSprop, Adam

# 字体设置
# font = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc')
# matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei']
# matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("load.xlsx")
df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'])

X = df.iloc[:, 5:10].values
X = X.astype(np.float64)

minMaxScaler_x = preprocessing.MinMaxScaler().fit(X[:25000, 0:4])
X[:, 0:4] = minMaxScaler_x.transform(X[:, 0:4])

minMaxScaler_y = preprocessing.MinMaxScaler().fit(X[:25000, 4:])
X[:, 4:] = minMaxScaler_y.transform(X[:, 4:])

def mape(true, pred):
    diff = np.abs(true - pred)
    return np.mean(diff / true)


def generator(data, lookback, delay, min_index, max_index,
    shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][4]
        yield samples, targets


lookback = 1440
step = 1
delay = 960
batch_size = 128

train_gen = generator(X,
                        lookback=lookback,
                        delay=delay,
                        min_index=0,
                        max_index=25000,
                        shuffle=True,
                        step=step,
                        batch_size=batch_size)
val_gen = generator(X,
                        lookback=lookback,
                        delay=delay,
                        min_index=25000,
                        max_index=30000,
                        step=step,
                        batch_size=batch_size)
test_gen = generator(X,
                        lookback=lookback,
                        delay=delay,
                        min_index=10944,
                        max_index=28444,
                        step=step,
                        batch_size=batch_size)

val_steps = (30000-25000-lookback) // batch_size

test_steps = (28444-10945-lookback) // batch_size

model = Sequential()
model.add(layers.LSTM(64, input_shape=(None, X.shape[-1]), return_sequences=False))
# model.add(layers.LSTM(64, return_sequences=False))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae','mape'])


history = model.fit_generator(train_gen,
                                steps_per_epoch=200,
                                epochs=20,
                                validation_data=val_gen,
                                validation_steps=val_steps)
#
# posibility = model.predict_generator(test_gen, steps=test_steps)
# print("posibility: ", posibility)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
# plt.savefig("/home/hu/PycharmProjects/bigpaper/figs/pv_loss.png")
plt.show()

oneMonth = 720
test_415 = np.zeros((oneMonth, lookback, 5))
true_415 = np.zeros((oneMonth, 1))
start_index = 10945

for i in range(oneMonth):
    test_415[i] = X[(start_index-lookback):start_index]
    true_415[i] = X[start_index+delay, 4:]
    start_index = start_index + 1

res_415 = model.predict(test_415)

x = range(1, len(res_415) + 1)

true = minMaxScaler_y.inverse_transform(true_415)
predict = minMaxScaler_y.inverse_transform(res_415)


def deci(x):
    return format(x, '.2f')

print("真实值 : 预测值 : 绝对误差 : 相对误差")
for i in range(len(true)):
    print(deci(true[i, 0]), " : ", deci(predict[i, 0]), " : ",
          deci(np.abs(true[i, 0] - predict[i, 0])), " : ",
          deci(np.abs(true[i, 0] - predict[i, 0])/true[i, 0]))

mape = mape(true,predict)
print("平均相对误差： ", mape)

plt.figure()
plt.plot(x, true, 'g*', label='true')
plt.plot(x, predict, 'ro', label='true')
plt.show()
