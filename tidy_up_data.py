from keras.datasets import mnist
import matplotlib.pyplot as plt
import random

import utils
import config

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_data = []
test_data = []
for i in range(10):
    train_data.append(x_train[y_train==i])
    test_data.append(x_test[y_test==i])

utils.save_to_pickle(train_data,config.TRAIN_DATA_PATH)
utils.save_to_pickle(test_data, config.TEST_DATA_PATH)

#ckeck data
train_data = utils.load_pickle(config.TRAIN_DATA_PATH)

plt.figure(figsize=(5, 10))
for i in range(10):
    data_single = train_data[i]
    show_sample_index = random.sample(range(len(data_single)), 5)
    for j in range(5):
        plt.subplot(10, 5, 5 * i + j + 1)
        plt.imshow(data_single[show_sample_index[j]])
    title = '%s' % (i)
    plt.title(title)
plt.show()