import pandas as pd
import numpy as np
import awkward
import itertools
import copy

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

batch_size = 28
batch_sizeV = 6
nfiles = 27000
#nfiles = 110
epochs = 100


def generator(batch_size,nfiles):

  samples_per_file = 30
  number_of_batches = int(samples_per_file/batch_size)
  counter=0
  fcnt = 101

  df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_noSci_noProp_1cmBS/GNN.pi150GeV_100.pkl.gz')
  df["Label"]=df["Label"]/1000.
  nhits = len(df["Points"][0])
  samples_per_file = len(df)
  number_of_batches = int(samples_per_file/batch_size)
  while 1:

    try:
       np1 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,:200].sum(axis=-1)
       np2 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,200:].sum(axis=-1)
       np3=np.concatenate([np1.reshape(batch_size,nhits,1),
                           np2.reshape(batch_size,nhits,1)],axis=2)

       X_batch = {'points':np.array(df["Points"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32"),
                #'features':np.array(df["Features"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16"),
                'features':np2,
                'mask':np.array(df["Mask"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16").reshape(batch_size,nhits,1)}
       y_batch=np.array(df["Label"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32")
       counter += 1
       yield X_batch, y_batch
    except:
       print("Something went wrong with the data") 
       counter += 1

    #restart counter to yeild data in the next epoch as well
    if fcnt > nfiles :

        fcnt = 100
    if counter >= number_of_batches:
        counter = 0
        df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_noSci_noProp_1cmBS/GNN.pi150GeV_'+str(fcnt)+'.pkl.gz')
        df["Label"]=df["Label"]/1000.
        samples_per_file = len(df)
        number_of_batches = int(samples_per_file/batch_size)

        fcnt += 1

def val_generator(batch_size,nfiles):

  counter=0
  fcnt = nfiles +1
  df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_noSci_noProp_1cmBS/GNN.pi150GeV_'+str(fcnt)+'.pkl.gz')
  samples_per_file = len(df)
  number_of_batches = samples_per_file/batch_size
  df["Label"]=df["Label"]/1000.
  nhits = len(df["Points"][0])

  while 1:
    try: 
       np1 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,:200].sum(axis=-1)
       np2 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,200:].sum(axis=-1)
       np3=np.concatenate([np1.reshape(batch_size,nhits,1),
                           np2.reshape(batch_size,nhits,1)],axis=2)

        
       X_batch = {'points':np.array(df["Points"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32"),
                'features':np2,
                #'features':np.array(df["Features"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16"),
                'mask':np.array(df["Mask"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16").reshape(batch_size,nhits,1)}
       y_batch=np.array(df["Label"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32")
       counter += 1
       yield X_batch, y_batch
    except:
       print("Something went wrong with the data") 
       counter += 1

    #restart counter to yeild data in the next epoch as well
    if fcnt > nfiles+3000 :
        fcnt = nfiles +1
    if counter >= number_of_batches:
        counter = 0
        df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_noSci_noProp_1cmBS/GNN.pi150GeV_'+str(fcnt)+'.pkl.gz')
        df["Label"]=df["Label"]/1000.
        samples_per_file = len(df)
        number_of_batches = int(samples_per_file/batch_size)
        fcnt += 1


import tensorflow as tf
from tensorflow import keras
from tf_keras_model import get_particle_net, get_particle_net_lite


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
 try:
   for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
   logical_gpus = tf.config.experimental.list_logical_devices('GPU')
   print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
 except RuntimeError as e:
   print(e)


strategy = tf.distribute.MirroredStrategy()


#model_type = 'particle_net_lite' # choose between 'particle_net' and 'particle_net_lite'
model_type = 'particle_net' # choose between 'particle_net' and 'particle_net_lite'
num_classes = 1

input_shapes = {'points': (nhits, 3), 'features': (nhits, 1), 'mask': (nhits, 1)}

if 'lite' in model_type:
    with strategy.scope():
        model = get_particle_net_lite(num_classes, input_shapes)
else:
    with strategy.scope():
        model = get_particle_net(num_classes, input_shapes)


# Training parameters
# batch_size = 1024 if 'lite' in model_type else 384
#batch_size = 100 if 'lite' in model_type else 16

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 5:
        lr *= 0.1
    elif epoch > 10:
        lr *= 0.01
    elif epoch > 15:
        lr *= 0.002
    logging.info('Learning rate: %f'%lr)
    return lr

model.compile(loss='mean_squared_logarithmic_error',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)) )
#              optimizer=keras.optimizers.Adam() )
model.summary()


# Prepare model model saving directory.
import os
save_dir = 'model_checkpoints'
model_name = 'GNN_3D_3cm_50ps_NoProp_2C_1cmBS.loss_{val_loss:01.6f}.e{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger()

early = keras.callbacks.EarlyStopping(monitor="val_loss",
                      mode="min", patience=16)

#callbacks = [checkpoint]
#callbacks = [checkpoint, lr_scheduler,early]
callbacks = [checkpoint,early]

# callbacks = [checkpoint, lr_scheduler, progress_bar]





model.fit(
          generator(batch_size,nfiles),
          steps_per_epoch = (nfiles-100)*(30/batch_size),
          epochs=epochs,
          validation_data=val_generator(batch_sizeV,nfiles),
          validation_steps=3000*int(30/batch_sizeV),
          callbacks=callbacks
          ,use_multiprocessing=True, workers=4, max_queue_size=240
         )
