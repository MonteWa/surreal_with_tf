import tensorflow as tf
import numpy as np
from hourglass_tiny import HourglassModel
import ConfigParser
import os
import pandas as pd
from keras.preprocessing import image
import scipy.io
from scipy.misc import imresize
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# load data(we need the same inputs but (None,256,256,15) label)
data_path = os.path.join('Sitting','img')
label_path = os.path.join('Sitting','masks')
data_names = sorted(os.listdir(data_path))
label_names = sorted(os.listdir(label_path))
data_addr = []
for i in range(len(data_names)):
    data_addr.append(os.path.join('Sitting/img',data_names[i]))

label_addr = []
for i in range(len(label_names)):
    label_addr.append(os.path.join('Sitting/masks',label_names[i]))
The_Main_List = pd.DataFrame(data_names,columns = ['data_name'])
temp_frame1 = pd.DataFrame(label_names,columns=['label_name'])
temp_frame2 = pd.DataFrame(data_addr,columns=['data_addr'])
temp_frame3 = pd.DataFrame(label_addr,columns=['label_addr'])
The_Main_List = pd.concat([The_Main_List,temp_frame1],axis=1)
The_Main_List = pd.concat([The_Main_List,temp_frame2],axis=1)
The_Main_List = pd.concat([The_Main_List,temp_frame3],axis=1)
print(The_Main_List.head())

# load one image
def load_image(path):
    img = image.load_img(path,target_size=(256,256))
    img = image.img_to_array(img)
    return img

# load one label for upsampling(label size:(None,256,256,15))
def load_label_as_gtmap(path):
    # load one label and resize it to [64,64]
    label = scipy.io.loadmat(path)
    label = label['M']
    label_resize = imresize(label,[256,256],interp='nearest')
    label_resize = label_resize.astype('float32')
    label_floor = np.floor(label_resize/18)
    # transform to sigle map
    map = np.zeros([256,256,15])
    for i in range(15):
        for n in range(256):
            for m in range(256):
                if label_floor[n][m] == i:
                    map[n,m,i] = 1.0
    return map

# load all data
data = np.zeros([len(The_Main_List),256,256,3])
for i,path in enumerate(The_Main_List.data_addr):
    data[i] = load_image(path)

# load all labels
label = np.zeros([len(The_Main_List),256,256,15])
for i,path in tqdm(enumerate(The_Main_List.label_addr)):
    label[i] = load_label_as_gtmap(path)


# split train and test,first 80% train,last 20% test.
rnd = np.arange(len(data))
train_idx = rnd < (0.8*len(data))
valid_idx = rnd >= (0.8*len(data))
train_data = data[train_idx]
test_data = data[valid_idx]
train_label = label[train_idx]
test_label = label[valid_idx]

def next_batch(data,label,batch_size):
    batch_x = np.zeros([batch_size,256,256,3])
    batch_y = np.zeros([batch_size,256,256,15])
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    for i in range(batch_size):
        batch_x[i] = data[idx[i]]
        batch_y[i] = label[idx[i]]
    return batch_x,batch_y

# build the model
def process_config(conf_file):
	"""
	"""
	params = {}
	config = ConfigParser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params
print('--Parsing Config File')
params = process_config('config.cfg')

def weight_variable(shape, stddev=0.02, name=None): #Create tensorflow matrix with  normal random distubotion mean 0 and standart deviation 0.02
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias): #Simple conv and biase addition this function is waste of time replace in the code with the tensorflow command
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME") #Padding same mean the output is same size as input?
    return tf.nn.bias_add(conv, bias)

model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'],
                       nModules=params['nmodules'],
                       nLow=params['nlow'], outputDim=params['num_joints'],
                       batch_size=params['batch_size'], attention = params['mcam'],
                       training=True, drop_rate= params['dropout_rate'],
                       lear_rate=params['learning_rate'],
                       decay=params['learning_rate_decay'],
                       decay_step=params['decay_step'],
                       name=params['name'],
                       logdir_train=params['log_dir_train'],
                       logdir_test=params['log_dir_test'],
                       tiny= params['tiny'], w_loss=params['weighted_loss'] ,
                       joints= params['joint_list'],modif=False)
model.generate_model()

# get the variables to restore
variables_to_restore = tf.contrib.slim.get_variables_to_restore()
# last map: (None,64,64,15)
output_stack = model.output
stacks = tf.unstack(output_stack,axis=1)
last_map = stacks[3]
# add layers for upsampling, from (64,64,15) to (256,256,15)
bi_upsampling = tf.image.resize_bilinear(last_map,[256,256],name='bilinear_upsampling')
W_last = weight_variable([3,3,15,15],name='W_last')
b_last = bias_variable([15],name='b_last')
aft_upsam_out = conv2d_basic(bi_upsampling,W_last,b_last)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=aft_upsam_out, labels= model.hr_label),
                      name= 'cross_entropy_loss')

train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# restore the hourglass net
saver = tf.train.Saver(variables_to_restore)
path = "my_net"
full_path = os.path.join(path,'save_net.ckpt')
saver.restore(sess,full_path)

# training
for i in range(2000):
    x, y = next_batch(train_data, train_label, 3)
    if i/50 == 0:
        print(sess.run(loss,feed_dict={model.img:x, model.hr_label:y}))
    sess.run(train_step, feed_dict={model.img: x, model.hr_label: y})

# save the network
path = "my_upsampling_net"
is_exist = os.path.exists(path)
if not is_exist:
    os.makedirs(path)
full_path = os.path.join(path,'save_net.ckpt')
saver2 = tf.train.Saver()
save_path = saver2.save(sess,full_path)