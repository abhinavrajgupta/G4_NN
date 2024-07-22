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

  df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_3cm_50ps_NoProp/GNN.pi150GeV_100.pkl.gz')
  df["Label"]=df["Label"]/1000.
  nhits = len(df["Points"][0])
  samples_per_file = len(df)
  number_of_batches = int(samples_per_file/batch_size)
  while 1:

    try:
       np1 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,:1].sum(axis=-1)
       np2 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,1:].sum(axis=-1)
       np3=np.concatenate([np1.reshape(batch_size,nhits,1),
                           np2.reshape(batch_size,nhits,1)],axis=2)

       X_batch = {'points':np.array(df["Points"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32"),
                #'features':np.array(df["Features"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16"),
                'features':np2,
                'mask':np.array(df["Mask"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16").reshape(batch_size,nhits,1)}
       y_batch=np.array(df["Label"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32")
       counter += 1
       points = X_batch['points']
       energies = X_batch['features']
       X_b = []
       for evt in range(0,len(points)):
           point=points[evt]
           energy=energies[evt]
           var = np.zeros((200,35,35),dtype=np.float32)
           for ihit in range(0,len(point)):
               hitP=point[ihit]
               hitE=energy[ihit]
               X=int(hitP[0]+49)
               Y=int(hitP[1]+15)
               Z=int(hitP[2]+10)
               if X>-1 and Y>-1 and Z>-1 and  Z<35 and Y<35 and X<200: var[X][Y][Z]+=hitE
               #else: 
               #    if(y_batch[evt]>8.): print("---  ",y_batch[evt], X,Y,Z,hitE)
           #R = np.sum(var)/np.sum(energies[evt])
           #if(y_batch[evt]>8.): print(y_batch[evt],R)
           #print(var.shape)
           X_b.append(var)
       X_b = np.asarray(X_b).reshape(len(points), 200, 35, 35, 1)
       #print(X_b.shape)
       #print(y_batch.shape)
       #yield X_batch, y_batch
       X_b_sum = X_b.sum(1).sum(1).sum(1)[:,0]
       yield [X_b,X_b_sum], y_batch
    except:
       print("Something went wrong with the data") 
       counter += 1

    #restart counter to yeild data in the next epoch as well
    if fcnt > nfiles :

        fcnt = 100
    if counter >= number_of_batches:
        counter = 0
        df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_3cm_50ps_NoProp/GNN.pi150GeV_'+str(fcnt)+'.pkl.gz')
        df["Label"]=df["Label"]/1000.
        samples_per_file = len(df)
        number_of_batches = int(samples_per_file/batch_size)

        fcnt += 1

def val_generator(batch_size,nfiles):

  counter=0
  fcnt = nfiles +1
  df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_3cm_50ps_NoProp/GNN.pi150GeV_'+str(fcnt)+'.pkl.gz')
  nhits = len(df["Points"][0])

  samples_per_file = len(df)
  number_of_batches = samples_per_file/batch_size
  df["Label"]=df["Label"]/1000.


  while 1:
    try: 
       np1 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,:1].sum(axis=-1)
       np2 = np.array(df["Features"].to_list(),dtype="int16")[batch_size*counter:batch_size*(counter+1),:,1:].sum(axis=-1)
       np3=np.concatenate([np1.reshape(batch_size,nhits,1),
                           np2.reshape(batch_size,nhits,1)],axis=2)

       X_batch = {'points':np.array(df["Points"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32"),
                'features':np2,
                #'features':np.array(df["Features"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16"),
                'mask':np.array(df["Mask"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="int16").reshape(batch_size,nhits,1)}
       y_batch=np.array(df["Label"][batch_size*counter:batch_size*(counter+1)].to_list(),dtype="float32")
       counter += 1
       points = X_batch['points']
       energies = X_batch['features']
       X_b = []
       for evt in range(0,len(points)):
           point=points[evt]
           energy=energies[evt]
           var = np.zeros((200,35,35),dtype=np.float32)
           for ihit in range(0,len(point)):
               hitP=point[ihit]
               hitE=energy[ihit]
               X=int(hitP[0]+49)
               Y=int(hitP[1]+15)
               Z=int(hitP[2]+10)
               if X>-1 and Y>-1 and Z>-1 and  Z<35 and Y<35 and X<200: var[X][Y][Z]+=hitE
               #else: 
               #    if(y_batch[evt]>8.): print("---  ",y_batch[evt], X,Y,Z,hitE)
           #R = np.sum(var)/np.sum(energies[evt])
           #if(y_batch[evt]>8.): print(y_batch[evt],R)
           #print(var.shape)
           X_b.append(var)
       X_b = np.asarray(X_b).reshape(len(points), 200, 35, 35, 1)
       #print(X_b.shape)
       #print(y_batch.shape)
       X_b_sum = X_b.sum(1).sum(1).sum(1)[:,0]
       yield [X_b,X_b_sum], y_batch
    except:
       print("Something went wrong with the data") 
       counter += 1

    #restart counter to yeild data in the next epoch as well
    if fcnt > nfiles+3000 :
        fcnt = nfiles +1
    if counter >= number_of_batches:
        counter = 0
        df = pd.read_pickle('/lustre/research/hep/jdamgov/idea_ntpl_v1/IDEA_pi_pkl3_3D_3cm_50ps_NoProp/GNN.pi150GeV_'+str(fcnt)+'.pkl.gz')
        df["Label"]=df["Label"]/1000.
        samples_per_file = len(df)
        number_of_batches = int(samples_per_file/batch_size)
        fcnt += 1


import tensorflow as tf
from tensorflow import keras
#from tf_keras_model import get_particle_net, get_particle_net_lite


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
 try:
   for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
   logical_gpus = tf.config.experimental.list_logical_devices('GPU')
   print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
 except RuntimeError as e:
   print(e)


#strategy = tf.distribute.MirroredStrategy()

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():

  input_layer1 = tf.keras.layers.Input((200, 35, 35, 1))
  input_layer2 = tf.keras.layers.Input((1))

  conv_layer1 = tf.keras.layers.Conv3D(64, kernel_size=(7,3,3) , activation='relu')(input_layer1)
  conv_layer2 = tf.keras.layers.Conv3D(32, kernel_size=(5,3,3) , activation='relu')(conv_layer1)
  pooling_layer1 = tf.keras.layers.MaxPool3D()(conv_layer2)
  conv_layer3 = tf.keras.layers.Conv3D(32, kernel_size=(5,3,3) , activation='relu')(pooling_layer1)
  conv_layer4 = tf.keras.layers.Conv3D(32, kernel_size=(5,3,3) , activation='relu')(conv_layer3)
  bnorm_layer2 = tf.keras.layers.BatchNormalization()(conv_layer4)
  pooling_layer2 = tf.keras.layers.MaxPool3D()(bnorm_layer2)
  conv_layer5 = tf.keras.layers.Conv3D(32, 3 , activation='relu')(pooling_layer2)
  conv_layer6 = tf.keras.layers.Conv3D(32, 3 , activation='relu')(conv_layer5)
  conv_layer7 = tf.keras.layers.Conv3D(8, kernel_size=(5,1,1) , activation='relu')(conv_layer6)
  #pooling_layer3 = tf.keras.layers.MaxPool3D()(conv_layer6)
  #flatn_layer = tf.keras.layers.Flatten()(pooling_layer3)
  flatn_layer = tf.keras.layers.Flatten()(conv_layer7)
  conc_layer = tf.keras.layers.Concatenate(axis=1)([flatn_layer,input_layer2])

  #dense_layer1 = tf.keras.layers.Dense(units=1024, activation='relu')(conc_layer)
  dense_layer2 = tf.keras.layers.Dense(units=768, activation='relu')(conc_layer)
  dense_layer2 = tf.keras.layers.Dropout(0.3)(dense_layer2)
  dense_layer3 = tf.keras.layers.Dense(units=128, activation='relu')(dense_layer2)
  dense_layer4 = tf.keras.layers.Dense(units=32, activation='relu')(dense_layer3)
  dense_layer4 = tf.keras.layers.Dropout(0.2)(dense_layer4)
  output_layer = tf.keras.layers.Dense(units=1, activation='linear')(dense_layer4)

  model = tf.keras.Model(inputs=[input_layer1,input_layer2], outputs=output_layer)


model.compile(loss='mean_squared_logarithmic_error',
              optimizer=keras.optimizers.Adam() )
model.summary()


# Prepare model model saving directory.
import os
save_dir = 'model_checkpoints'
model_name = 'CNN_3D_3cm_50ps_3D_noProp_1C_more0ph.loss_{val_loss:01.6f}.e{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

progress_bar = keras.callbacks.ProgbarLogger()

early = keras.callbacks.EarlyStopping(monitor="val_loss",
                      mode="min", patience=12)

callbacks = [checkpoint,early]






model.fit(
          generator(batch_size,nfiles),
          steps_per_epoch = (nfiles-100)*(30/batch_size),
          epochs=epochs,
          validation_data=val_generator(batch_sizeV,nfiles),
          validation_steps=3000*int(30/batch_sizeV),
          callbacks=callbacks
          ,use_multiprocessing=True, workers=4, max_queue_size=60
         )
