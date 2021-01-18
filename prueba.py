#from src.datasets import brixiascore_cohen  as bsc
from src.datasets import brixiascore
from src.datasets import lung_segmentation
from src.BSNet.model import BSNet

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Check the docsting for additional info

DSBrixiascore = brixiascore.get_data_iterator(file_csv="data/metadata_global_v2.csv",
             data_directory="data/dicom_clean",preprocessing=True)
DSSegmentationTrain, DSSegmentationVal  = lung_segmentation.get_data()



#for elem in DS.as_numpy_iterator():
#    print(elem[0].shape)

models = BSNet(load_seg_model=False,load_align_model=True,load_bscore_model=False,freeze_segmentation=False,freeze_encoder=False)

#print(models[0].summary())
#print(models[1].summary())
#print(models[2].summary())

segmentation_model = models[0]

'''
    for elem in DSSegmentationTrain:
        plt.imshow(elem[0][0,:,:,0])
        plt.show()
        plt.imshow(elem[1][0,:,:,0])
        plt.show()
        predicted = segmentation_model.predict(elem[0])
        plt.imshow(predicted[0,:,:,0])
        plt.show()
'''
adam = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')


segmentation_model.compile(optimizer=adam,loss= tf.keras.losses.BinaryCrossentropy())

segmentation_model.fit(x=DSSegmentationTrain,steps_per_epoch=5,validation_data=DSSegmentationVal,validation_steps=2,epochs=10,batch_size=16)#,validation_data=DSSegmentationVal)


