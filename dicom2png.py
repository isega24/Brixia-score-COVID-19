import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from src.datasets.utils import equalize

img_height, img_width = 512, 512

dicom_clean = "data/dicom_clean/"

output = "data/png_resized/"

if not os.path.isdir(output):
    os.mkdir(output)

def decode_img(img_path):
    image_bytes = tf.io.read_file(img_path)
    img = tfio.image.decode_dicom_image(image_bytes)
    img = (img-tf.reduce_min(img))/(tf.reduce_max(img)-tf.reduce_min(img))
    img = equalize(img)*256
    return tf.image.resize(img, [img_height, img_width])

for file in tqdm(os.listdir(dicom_clean)):
    #if not os.path.exists((output+file).replace(".dcm",".png")):
    image = decode_img(dicom_clean+file)
    plt.imsave((output+file).replace(".dcm",".png"),image[0,:,:,0],cmap="gray")
