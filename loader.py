import os, math, pathlib, glob, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class DatasetFromDirectory:

    def __init__(self, train_data_dir, val_data_dir, 
                    image_size=224, batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_data_dir = [os.path.join(train_data_dir,"train/anchor","*.*"),
                                os.path.join(train_data_dir,"train/sim","*.*"),
                                os.path.join(train_data_dir,"train/diff","*.*")]
        self.val_data_dir = [os.path.join(val_data_dir,"val/anchor","*.*"),
                                os.path.join(val_data_dir,"val/sim","*.*"),
                                os.path.join(val_data_dir,"val/diff","*.*")]

    def _decode(self, path1, path2, path3):
        anchor = tf.io.read_file(path1)
        anchor = tf.io.decode_jpeg(anchor)
        anchor = tf.cast(anchor,tf.float32)/255.
        anchor = tf.image.resize(anchor, [self.image_size, self.image_size], 'bilinear')
        sim = tf.io.read_file(path2)
        sim = tf.io.decode_jpeg(sim)
        sim = tf.cast(sim,tf.float32)/255.
        sim = tf.image.resize(sim, [self.image_size, self.image_size], 'bilinear')
        diff = tf.io.read_file(path3)
        diff = tf.io.decode_jpeg(diff)
        diff = tf.cast(diff,tf.float32)/255.
        diff = tf.image.resize(diff, [self.image_size, self.image_size], 'bilinear')
        #label_sim = tf.strings.split(path2,"/")[-1]
        #label_sim = tf.strings.split(label_sim,"_")[-1]
        #label_sim = tf.strings.join([tf.strings.split(label_sim,".")[0],tf.strings.split(label_sim,".")[1]],".")
        #label_sim = tf.strings.to_number(label_sim)
        label_sim = 1.0
        label_diff = 0.0
        return (anchor, sim, diff, label_sim, label_diff)

    def get(self):
        train_files_anchor = tf.data.Dataset.list_files(self.train_data_dir[0],shuffle=False)
        train_files_sim = tf.data.Dataset.list_files(self.train_data_dir[1],shuffle=False)
        train_files_diff = tf.data.Dataset.list_files(self.train_data_dir[2],shuffle=False)
        train_ds = tf.data.Dataset.zip((train_files_anchor,train_files_sim,train_files_diff)).map(self._decode,self.AUTOTUNE).batch(self.batch_size).prefetch(self.AUTOTUNE)
        val_files_anchor = tf.data.Dataset.list_files(self.val_data_dir[0],shuffle=False)
        val_files_sim = tf.data.Dataset.list_files(self.val_data_dir[1],shuffle=False)
        val_files_diff = tf.data.Dataset.list_files(self.val_data_dir[2],shuffle=False)
        val_ds = tf.data.Dataset.zip((val_files_anchor,val_files_sim,val_files_diff)).map(self._decode,self.AUTOTUNE).batch(self.batch_size).prefetch(self.AUTOTUNE)
        return train_ds, val_ds