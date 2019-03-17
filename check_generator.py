import utils

import matplotlib.pyplot as plt

train_gen = utils.data_generator(20)

x_batch,y_batch = next(train_gen)

print(x_batch[0].shape, x_batch[1].shape, y_batch.shape)

fig = plt.figure(figsize=(2,50))

for i in range(20):
    plt.subplot(20, 2, 2 * i + 1)
    plt.imshow(x_batch[0][i].reshape([28,28]))
    plt.subplot(20, 2, 2 * i + 2)
    plt.imshow(x_batch[1][i].reshape([28,28]))

    plt.title(str(y_batch[i]))

fig.tight_layout()
plt.show()
