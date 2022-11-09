import sys, os, cv2, math ,time, gc, glob, random
from shutil import move, copyfile, copy2
sys.path.append('/home/kevin/ML')
sys.path.append('/home/kevin/ML/similarity-cnn')
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import layers
import tensorflow_addons as tfa
import efficientnet
from models.efficientnet import efficientnet as eff_full
from loader import DatasetFromDirectory
import model_design
from datetime import datetime

def cdist_metrics(label_sim, label_diff, y_pred_sim_anchor, y_pred_sim_other, y_pred_diff_anchor, y_pred_diff_other):
    cdist_sim = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(y_pred_sim_anchor, 1) , tf.nn.l2_normalize(y_pred_sim_other, 1) ), 1)
    cdist_diff = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(y_pred_diff_anchor, 1) , tf.nn.l2_normalize(y_pred_diff_other, 1) ), 1)
    cdist_error = tf.reduce_mean(tf.abs(label_sim-cdist_sim) + tf.abs(label_diff-cdist_diff))
    cdist_acc = (tf.keras.metrics.binary_accuracy(label_sim, cdist_sim, 0.5) + tf.keras.metrics.binary_accuracy(label_diff, cdist_diff, 0.5))/2
    return cdist_error, cdist_acc

def get_model1():

    inputs_A = tf.keras.Input([224,224,3])
    inputs_B = tf.keras.Input([224,224,3])
    # A = tf.squeeze(inputs_A,1)
    # B = tf.squeeze(inputs_B,1)
    _, outputs_A = efficientnet.EfficientNet(inputs_A,1.0,1.0,"A",0,0.2,"bn",True,8)
    _, outputs_B = efficientnet.EfficientNet(inputs_B,1.0,1.0,"B",0,0.2,"bn",True,8)

    def abs_diff(x):
        A,B = x
        y = tf.math.abs(A-B)
        return y

    x = tf.keras.layers.Concatenate()([outputs_A,outputs_B])
    #x = tf.keras.layers.Lambda(abs_diff)([outputs_A,outputs_B])
    #x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1,activation="sigmoid", dtype=tf.float32)(x)
    model = tf.keras.Model([inputs_A,inputs_B],outputs)
    return model

def get_model():

    eff_in = tf.keras.Input([224,224,3])
    x = tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1,0.1),(-0.1,0.1))(eff_in)
    x = tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation((-0.1,0.1))(x)
    x = tf.keras.layers.GaussianNoise(0.1)(x)
    #_, eff_out = efficientnet.EfficientNet(x,1.0,1.0,None,0.2,0.2,"bn",True,"mish",8)
    _, eff_out = eff_full.EfficientNet(x,1.0,1.0,None,0.2,0.2,"bn",True,"mish",8)
    eff = tf.keras.Model(eff_in,eff_out)
    #eff = efficientnet.load_weights(eff,"efficientnet-b0","noisy-student")

    inputs_A = tf.keras.Input([224,224,3],name="inputs_A")
    inputs_B = tf.keras.Input([224,224,3],name="inputs_B")

    A = eff(inputs_A)
    B = eff(inputs_B)

    outputs_A = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), dtype=tf.float32)(A)
    outputs_B = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), dtype=tf.float32)(B)
    model = tf.keras.Model([inputs_A,inputs_B],[outputs_A,outputs_B])
    return model

class Eff_Model(tf.keras.Model):

    def __init__(self):
        super(Eff_Model,self).__init__()
        pass

    def call(self, inputs):

        inputs_A, inputs_B = inputs
        _, outputs_A = efficientnet.EfficientNet(inputs_A,1.0,1.0,"A",0.2,0.2,"bn",True,8)
        _, outputs_B = efficientnet.EfficientNet(inputs_B,1.0,1.0,"B",0.2,0.2,"bn",True,8)

        def abs_diff(x):
            A,B = x
            x = tf.math.abs(A-B)
            return x
            
        x = tf.keras.layers.Lambda(abs_diff)([outputs_A,outputs_B])
        outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        #outputs = tf.keras.layers.Dense(1,activation="sigmoid")(x)
        return outputs

@tf.function
def train_step_fn(anchor,sim,diff,label_sim,label_diff):

    def triplet_loss(y_pred_sim_anchor,y_pred_sim_other,y_pred_diff_anchor,y_pred_diff_other):
        d_pos = tf.reduce_sum(tf.square(y_pred_sim_anchor - y_pred_sim_other), 1)
        d_neg = tf.reduce_sum(tf.square(y_pred_diff_anchor - y_pred_diff_other), 1)
        loss = tf.maximum(0.0, triplet_margin + d_pos - d_neg)
        loss = tf.reduce_mean(loss)
        return loss

    def tf_triplet_loss(y_true,y_pred):
        return tfa.losses.triplet_semihard_loss(y_true,y_pred,triplet_margin)

    def contrastive_loss(y_true,y_pred_anchor,y_pred_other):
        d = tf.reduce_sum(tf.square(y_pred_anchor - y_pred_other), 1)
        d_sqrt = tf.sqrt(d)
        loss = (1-y_true) * tf.square(tf.maximum(0., contrastive_margin - d_sqrt)) + y_true * d
        return 0.5 * tf.reduce_mean(loss)

    def bce(y_true,y_pred):
        return - y_true*tf.math.log(y_pred+epsilon) - (1-y_true)*tf.math.log(1-y_pred+epsilon)

    with tf.GradientTape() as tape:
        y_pred_sim_anchor, y_pred_sim_other = model([anchor, sim], training=True)
        #cur_train_loss = contrastive_loss(label_sim,y_pred_sim_anchor,y_pred_sim_other)
        y_pred_diff_anchor, y_pred_diff_other = model([anchor, diff], training=True)
        #cur_train_loss += contrastive_loss(label_diff,y_pred_diff_anchor,y_pred_diff_other)
        cur_train_loss = triplet_loss(y_pred_sim_anchor,y_pred_sim_other,y_pred_diff_anchor,y_pred_diff_other)
        scaled_loss = opt.get_scaled_loss(cur_train_loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = opt.get_unscaled_gradients(scaled_gradients)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return cur_train_loss, y_pred_sim_anchor, y_pred_sim_other, y_pred_diff_anchor, y_pred_diff_other

#@tf.function(experimental_compile=True)
@tf.function
def val_step_fn(anchor,sim,diff,label_sim,label_diff):

    def triplet_loss(y_pred_sim_anchor,y_pred_sim_other,y_pred_diff_anchor,y_pred_diff_other):
        d_pos = tf.reduce_sum(tf.square(y_pred_sim_anchor - y_pred_sim_other), 1)
        d_neg = tf.reduce_sum(tf.square(y_pred_diff_anchor - y_pred_diff_other), 1)
        loss = tf.maximum(0.0, triplet_margin + d_pos - d_neg)
        loss = tf.reduce_mean(loss)
        return loss

    def tf_triplet_loss(y_true,y_pred):
        return tfa.losses.triplet_semihard_loss(y_true,y_pred,triplet_margin)

    def contrastive_loss(y_true,y_pred_anchor,y_pred_other):
        d = tf.reduce_sum(tf.square(y_pred_anchor - y_pred_other), 1)
        d_sqrt = tf.sqrt(d)
        loss = (1-y_true) * tf.square(tf.maximum(0., contrastive_margin - d_sqrt)) + y_true * d
        return 0.5 * tf.reduce_mean(loss)

    y_pred_sim_anchor, y_pred_sim_other = model([anchor, sim], training=False)
    #cur_val_loss = contrastive_loss(label_sim,y_pred_sim_anchor,y_pred_sim_other)
    y_pred_diff_anchor, y_pred_diff_other = model([anchor, diff], training=False)
    #cur_val_loss += contrastive_loss(label_diff,y_pred_diff_anchor,y_pred_diff_other)
    cur_val_loss = triplet_loss(y_pred_sim_anchor,y_pred_sim_other,y_pred_diff_anchor,y_pred_diff_other)
    # for i in metrics_list:
    #     i.update_state(label_sim, y_pred_sim_anchor, y_pred_sim_other)
    #     i.update_state(label_diff, y_pred_diff_anchor, y_pred_diff_other)
    return cur_val_loss, y_pred_sim_anchor, y_pred_sim_other, y_pred_diff_anchor, y_pred_diff_other

def linear_warmup_cosine_decay(epoch):
    k = 2
    steps = 5
    lowest_lr_on_epoch = 85 / (math.pi/2)
    lowest_lr = lr_init*k/100
    current_lr = tf.keras.backend.get_value(opt.learning_rate)
    if epoch == 1:
        lr = current_lr
    elif epoch > 1 and epoch <= 1+steps:
        lr = current_lr+lr_init*(k-1)/steps
    else: # epoch > steps:
        lr = (lr_init*k-lowest_lr) *math.cos((epoch-steps)/lowest_lr_on_epoch)**2 + lowest_lr
    if epoch > steps+1 and lr > current_lr:
        lr = current_lr
    return lr

mode = 'train' ################## change mode here ######################

if mode == 'train':

    img_size = 224
    batch_size = 16
    num_class = 1
    lr_init = 0.001
    epsilon = 1e-8
    total_epoch = 100
    triplet_margin = tf.constant(1.0)
    contrastive_margin = tf.constant(1.0)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    opt = tfa.optimizers.NovoGrad(learning_rate=lr_init,weight_decay=0.0005,epsilon=epsilon)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = get_model()
    #model = model_design.get(model_design.sub_model1(img_size),img_size)
    model.load_weights("w_b0/1.h5")
    print(model.summary())
    ds = DatasetFromDirectory("/home/kevin/ML/dataset/similarity","/home/kevin/ML/dataset/similarity",img_size,batch_size)
    train_ds, val_ds = ds.get()

    ####### custom train #########

    cur_train_loss, cur_val_loss = tf.constant(0.,shape=(batch_size,1)), tf.constant(0.,shape=(batch_size,1))
    for cur_epoch in range(1,total_epoch+1):
        start_time = time.time()
        tf.keras.backend.set_value(opt.learning_rate, linear_warmup_cosine_decay(cur_epoch))   ###########  change LR schedule here ###########
        print('~~~ epoch %i/%i lr:%.5f ~~~'%(cur_epoch,total_epoch,tf.keras.backend.get_value(opt.learning_rate)))
        temp_train_loss = []
        cdist_error_list, cdist_acc_list = [], []
        for step, (anchor,sim,diff,label_sim,label_diff) in enumerate(train_ds,1):
            cur_train_loss, y_pred_sim_anchor, y_pred_sim_other, y_pred_diff_anchor, y_pred_diff_other = train_step_fn(anchor,sim,diff,label_sim,label_diff)
            temp_train_loss.append(tf.reduce_mean(cur_train_loss))
            cdist_error, cdist_acc = cdist_metrics(label_sim, label_diff, y_pred_sim_anchor, y_pred_sim_other, y_pred_diff_anchor, y_pred_diff_other)
            cdist_error_list.append(cdist_error.numpy()), cdist_acc_list.append(cdist_acc.numpy())
            print('train --- step: %i, loss: %.4f, error: %.4f, acc: %.4f'%(
                    step,np.mean(temp_train_loss),sum(cdist_error_list)/len(cdist_error_list),sum(cdist_acc_list)/len(cdist_acc_list)),end='\r')
        temp_val_loss = []
        cdist_error_list, cdist_acc_list = [], []
        for step, (anchor,sim,diff,label_sim,label_diff) in enumerate(val_ds,1):
            cur_val_loss, y_pred_sim_anchor, y_pred_sim_other, y_pred_diff_anchor, y_pred_diff_other = val_step_fn(anchor,sim,diff,label_sim,label_diff)
            temp_val_loss.append(tf.reduce_mean(cur_val_loss))
            cdist_error, cdist_acc = cdist_metrics(label_sim, label_diff, y_pred_sim_anchor, y_pred_sim_other, y_pred_diff_anchor, y_pred_diff_other)
            cdist_error_list.append(cdist_error.numpy()), cdist_acc_list.append(cdist_acc.numpy())
        print('\nval --- step: %i, loss: %.4f, error: %.4f, acc: %.4f'%(
                    step,np.mean(temp_val_loss),sum(cdist_error_list)/len(cdist_error_list),sum(cdist_acc_list)/len(cdist_acc_list)))
        model.save_weights('w_b0_slim2/ep%i-%.4f-%.4f-%.4f.h5'%(cur_epoch,np.mean(temp_val_loss),sum(cdist_error_list)/len(cdist_error_list),sum(cdist_acc_list)/len(cdist_acc_list)))
        print('time spent: %.1fs'%(time.time()-start_time))

