import tensorflow as tf
import numpy as np
from hourglass_tiny import HourglassModel
import ConfigParser
import os
import pandas as pd
from keras.preprocessing import image
import scipy.io
from scipy.misc import imresize


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

# load one label
def load_label_as_gtmap(path):
    # load one label and resize it to [64,64]
    label = scipy.io.loadmat(path)
    label = label['M']
    label_resize = imresize(label,[64,64],interp='nearest')
    label_resize = label_resize.astype('float32')
    label_floor = np.floor(label_resize/18)
    # transform to sigle map
    map = np.zeros([64,64,15])
    for i in range(15):
        for n in range(64):
            for m in range(64):
                if label_floor[n][m] == i:
                    map[n,m,i] = 1.0
    # transform to gtMap
    gtmap = np.zeros([4,64,64,15])
    for i in range(4):
        gtmap[i] = map
    return gtmap

# load all data
data = np.zeros([len(The_Main_List),256,256,3])
for i,path in enumerate(The_Main_List.data_addr):
    data[i] = load_image(path)
# load all labels
label = np.zeros([len(The_Main_List),4,64,64,15])
for i,path in enumerate(The_Main_List.label_addr):
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
    batch_y = np.zeros([batch_size,4,64,64,15])
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    for i in range(batch_size):
        batch_x[i] = data[idx[i]]
        batch_y[i] = label[idx[i]]
    return batch_x,batch_y

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

loss = model.loss
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#path = "my_net"
#full_path = os.path.join(path,'save_net.ckpt')
#saver.restore(sess,full_path)

# train
for i in range(2000):
    x, y = next_batch(train_data, train_label, 8)
    sess.run(train_step, feed_dict={model.img: x, model.gtMaps: y})
    if i%50 == 0:
        print(sess.run(model.loss,feed_dict={model.img:x, model.gtMaps:y}))

# save the network
path = "my_net"
is_exist = os.path.exists(path)
if not is_exist:
    os.makedirs(path)
full_path = os.path.join(path,'save_net.ckpt')
save_path = saver.save(sess,full_path)
