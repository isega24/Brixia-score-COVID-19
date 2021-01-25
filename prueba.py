#from src.datasets import brixiascore_cohen  as bsc
from src.datasets import brixiascore
from src.datasets import lung_segmentation
from src.BSNet.model import BSNet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Check the docsting for additional info

DSBrixiaTrain, DSBrixiaTest = brixiascore.get_data_iterator_png(file_csv="data/metadata_global_v2.csv",data_directory="data/png_resized",preprocessing=True)



#DSSegmentationTrain, DSSegmentationVal  = lung_segmentation.get_data()



#for elem in DS.as_numpy_iterator():
#    print(elem[0].shape)
segmentation_checkpoint = "./src/BSNet/weights/segmentation-model.h5"
brixia_checkpoint = './src/BSNet/weights/bscore-model.h5'

models = BSNet(seg_model_weights=segmentation_checkpoint,
    bscore_model_weights=brixia_checkpoint,
    load_align_model=True,
    load_seg_model=True,
    load_bscore_model=False,
    freeze_segmentation=False,
    freeze_encoder=False,
    freeze_align_model=False)


segmentation_model = models[0]
alignment_model = models[1]
brixia_model = models[2]

adam = tf.keras.optimizers.Adam(
    learning_rate=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="AdamW")

checkpoint_seg = tf.keras.callbacks.ModelCheckpoint(filepath=segmentation_checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)

checkpoint_brixia = tf.keras.callbacks.ModelCheckpoint(filepath=brixia_checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)


################ ENTRENAMIENTO DEL MODELO DE SEGMENTACION
# segmentation_model.compile(optimizer=adam,loss= tf.keras.losses.BinaryCrossentropy(from_logits=True)  ,metrics=["acc"])#,tf.keras.metrics.MeanIoU(num_classes=2) ])
# segmentation_model.fit(x=DSSegmentationTrain,epochs=50,callbacks=[checkpoint_seg],steps_per_epoch=564//8+1,validation_data=DSSegmentationVal,validation_steps=159//8+1)
# segmentation_model.evaluate(DSSegmentationVal,steps=159//8+1)

def custom_loss(y_true,y_pred):
    alpha=0.7
    y_true_expanded = tf.expand_dims(y_true,axis=-1)

    y_pred_sum = tf.multiply(y_pred,np.array([0,1,2,3]))
    calcY_pred = tf.expand_dims(tf.reduce_sum(y_pred_sum,axis=-1),axis=-1)

    LossMAE = tf.losses.mean_absolute_error(y_true_expanded,calcY_pred)
    
    lossCrossEntropy = tf.losses.sparse_categorical_crossentropy(y_true,y_pred )
    weightLossMAE = tf.multiply(1.-alpha, tf.cast(LossMAE,tf.float32))
    weightCrossEntropy = tf.multiply(lossCrossEntropy,alpha)
    return tf.add(weightLossMAE,weightCrossEntropy)

alpha=0.7

brixia_model.compile(optimizer=adam,loss=tf.keras.losses.SparseCategoricalCrossentropy() ,metrics=["acc"])#,tf.keras.metrics.MeanIoU(num_classes=2) ])
brixia_model.fit(x=DSBrixiaTrain,validation_data=DSBrixiaTest,steps_per_epoch = int(4698*0.8)//8, validation_steps = int(4698*0.2)//8,epochs=100,callbacks=[checkpoint_brixia])
brixia_model.evaluate(x=DSBrixiaTest,steps= int(4698*0.2)//8)

#for batch in DSBrixiaTest:
#    print(batch[0].shape,batch[1].shape)
#    prediction = np.argmax(brixia_model.predict(batch[0]),axis=-1)
#    for i in range(len(prediction)):
#        print(prediction[i],batch[1][i])