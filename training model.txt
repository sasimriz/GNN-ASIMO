@author: Asim
'''
from graph_nets import utils_tf
import tensorflow as tf
from graph_nets.demos import models
import sonnet as snt
from graph_nets import utils_np

from gene_samples import graph_feas_train, rest_feas_train,Ys_train
from gene_samples import graph_feas_test, rest_feas_test,Ys_test

import numpy as np

config = tf.ConfigProto()
 
# ??GPU??????,????,???
config.gpu_options.allow_growth = True
 
# ??????????
config.gpu_options.per_process_gpu_memory_fraction = 0.1

G_fea_ph=utils_tf.placeholders_from_networkxs(graph_feas_train, force_dynamic_num_graphs=True)

def mse_loss(y,yTrue):
    return tf.reduce_mean(tf.square(y-yTrue))

def mae_loss(y,yTrue):
    return tf.reduce_mean(tf.abs(y-yTrue))

def mape_loss(y,yTrue):
    return tf.reduce_mean(tf.abs((y/yTrue)-1))

def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]

def gene_feed_dict():
    train_G_feas=utils_np.networkxs_to_graphs_tuple(graph_feas_train)
    test_G_feas=utils_np.networkxs_to_graphs_tuple(graph_feas_test)
    train_rest_fea=rest_feas_train
    test_rest_fea=rest_feas_test
    trainY=Ys_train
    testY=Ys_test
    train_feed={G_fea_ph:train_G_feas,rest_fea_ph:train_rest_fea,yTrue:trainY}
    test_feed={G_fea_ph:test_G_feas,rest_fea_ph:test_rest_fea,yTrue:testY}
    return train_feed,test_feed
    
np.savetxt('trueY_0402.csv',np.array(Ys_test),delimiter=',', fmt='%f')
#G_fea_ph[0]=make_all_runnable_in_session([G_fea_ph])

G_fea_ph=utils_tf.make_runnable_in_session(G_fea_ph)

model = models.EncodeProcessDecode(edge_output_size=32, node_output_size=32,global_output_size= 32)

encoded_G = model(G_fea_ph, 2)[0].globals

print(encoded_G.shape)

rest_fea_ph=tf.placeholder(tf.float64, [None,12])

yTrue=tf.placeholder(tf.float64,[None,1])

M_fea=tf.concat([encoded_G,rest_fea_ph],axis=1)

print(M_fea.shape)

HiddenSize=[256]

layers=[]

for i in HiddenSize:
    hidden_i=snt.Linear(output_size=i)
    layers.append(hidden_i)
    layers.append(tf.nn.relu)

hiddent_to_out=snt.Linear(output_size=1)

layers.append(hiddent_to_out)

mlp=snt.Sequential(layers)

y = mlp(M_fea)

print(y.shape)

lossfun=mse_loss(y,yTrue)

train_step=tf.train.AdamOptimizer(learning_rate=0.15).minimize(lossfun)

train_feed,test_feed=gene_feed_dict()

def savePreY(preY,i):
    np.savetxt('preY_0402_'+str(i)+'.csv',preY,delimiter=',', fmt='%f')

with tf.Session(config = config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(int(2e3)):
        train_values = sess.run({"step": train_step,
          "loss": lossfun},feed_dict=train_feed)
        test_values = sess.run({
            "loss": lossfun,
            "outputs": y,
        },feed_dict=test_feed)
        print("train_loss",train_values["loss"],"test_loss",test_values["loss"])
        if i>=500 and i % 100 ==0:
            savePreY(test_values["outputs"],i)

