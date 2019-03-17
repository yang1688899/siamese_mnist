import keras as k
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import sgd
from keras.metrics import binary_accuracy as acc

import network
import utils
import config

model = network.build_model()
model.summary()
opt = sgd(lr=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=[acc])

#callback
early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(config.SAVE_MODEL_PATH, save_best_only=True,
                                   monitor='val_acc', mode='max',verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                      min_delta=0.005, mode='max', verbose=1)

train_gen = utils.data_generator(config.BATCH_SIZE)
val_gen = utils.data_generator(config.BATCH_SIZE,is_train=False)


hist = model.fit_generator(train_gen,
                           steps_per_epoch=1000,
                           epochs=1000,
                           validation_data=val_gen,
                           validation_steps=10,
                           callbacks =[early_stopping,model_checkpoint,reduce_lr],
                           verbose=2
)