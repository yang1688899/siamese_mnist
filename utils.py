from keras.datasets import mnist
import pickle
import random
import numpy as np

import config

def save_to_pickle(obj,savepath):
    with open(savepath,"wb") as file:
        pickle.dump(obj,file)

def load_pickle(path):
    with open(path,"rb") as file:
        obj = pickle.load(file)
        return obj



def data_generator(batch_size, is_train=True):
    if is_train:
        sample_data = load_pickle(config.TRAIN_DATA_PATH)
    else:
        sample_data = load_pickle(config.TEST_DATA_PATH)
    list_1 = []
    list_2 = []
    labels = []
    while True:
        for i in range(batch_size):
            #随机生成一个正样本对或负样本对
            if random.random()>0.5:
                cls = random.randint(0,9)
                train_sample = sample_data[cls]
                sample_indices = random.sample( range(len(train_sample)),2 )

                list_1.append(train_sample[sample_indices[0]])
                list_2.append(train_sample[sample_indices[1]])
                labels.append(1)
            else:
                cls_indices = random.sample( range(10),2 )
                train_sample_1 = sample_data[cls_indices[0]]
                train_sample_2 = sample_data[cls_indices[1]]
                sample_index_1 = random.randint(0,len(train_sample_1))
                sample_index_2 = random.randint(0,len(train_sample_2))

                list_1.append( train_sample_1[sample_index_1] )
                list_2.append( train_sample_2[sample_index_2] )
                labels.append(0)

        x_batch = [np.array(list_1).reshape(-1,28,28,1), np.array(list_2).reshape(-1,28,28,1)]
        y_batch = np.array(labels)
        yield x_batch,y_batch




