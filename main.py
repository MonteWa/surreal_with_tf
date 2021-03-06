from __future__ import division
import tensorflow as tf
import numpy as np
from hourglass_tiny import HourglassModel
import ConfigParser
import os
import pandas as pd
from keras.preprocessing import image
import scipy.io
from scipy.misc import imresize
import gc

# get the path of the data
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
print(The_Main_List.head(10))

# load one image
def load_image(path):
    img = image.load_img(path,target_size=(256,256))
    img = image.img_to_array(img)
    return img
# laod one label as mask
def load_mask(path):
    label = scipy.io.loadmat(path)
    label = label['M']
    label_resize = imresize(label,[64,64],interp='nearest')
    label_resize = label_resize.astype('float32')
    label_floor = np.floor(label_resize/18)
    return label_floor

# load all data
data = np.zeros([len(The_Main_List),256,256,3])
for i,path in enumerate(The_Main_List.data_addr):
    data[i] = load_image(path)
# load all labels
label = np.zeros([len(The_Main_List),64,64])
for i,path in enumerate(The_Main_List.label_addr):
    label[i] = load_mask(path)

# split train and test
np.random.seed(seed=1993)
rnd = np.arange(len(data))
train_idx = rnd < 180
valid_idx = rnd >= 180
train_data = data[train_idx]
test_data = data[valid_idx]
train_label = label
test_label = label[valid_idx]
# save some memory
del data
del label
gc.collect()

def augmentor_with_keras(img_asarr,mask):
    temp = np.zeros([256,256,4])
    temp[:,:,0:3] = img_asarr
    temp[128-32:128+32,128-32:128+32,3]= mask
    after_rot = image.random_rotation(temp,rg=20,
                                      row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    after_zoom = image.random_zoom(after_rot,zoom_range=[0.7,1.3],
                                   row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    # random_shift
    return after_zoom[:,:,0:3],after_zoom[128-32:128+32,128-32:128+32,3]

def transfer_mask_to_gtmap(mask):
    map = np.zeros([64,64,15])
    for i in range(15):
        for n in range(64):
            for m in range(64):
                if mask[n][m] == i:
                    map[n,m,i] = 1.0
    # transform to gtMap
    gtmap = np.zeros([8,64,64,15])
    for i in range(8):
        gtmap[i] = map
    return gtmap

def get_next_batch(imgs,masks):
    length = len(imgs)
    idx = np.random.randint(0,length)
    img = imgs[idx]
    mask = masks[idx]
    return img,mask

def get_augmentor_batch(img,mask,aug_level):
    gtmap = transfer_mask_to_gtmap(mask)
    x_batch = np.zeros([aug_level,256,256,3])
    y_batch = np.zeros([aug_level,8,64,64,15])
    x_batch[0]=img
    y_batch[0]=gtmap
    for i in range(aug_level-1):
        aug_img,aug_mask = augmentor_with_keras(img,mask)
        gtmap = transfer_mask_to_gtmap(aug_mask)
        x_batch[i+1] = aug_img
        y_batch[i+1] = gtmap
    return x_batch,y_batch

def cal_acc(pred_map,mask):
    length = mask.shape[0]
    wideth = mask.shape[1]
    pixel_acc_list = []
    for i in range(length):
        for j in range(wideth):
            if mask[i,j]!=0:
                if pred_map[i,j] == mask[i,j]:
                    pixel_acc_list.append(1)
                else:
                    pixel_acc_list.append(0)
    acc = np.mean(pixel_acc_list)
    return acc

def cal_iou(pred_map,mask):
    mask_matrix = np.zeros([64, 64, 15])
    for i in range(15):
        for n in range(64):
            for m in range(64):
                if mask[n][m] == i:
                    mask_matrix[n, m, i] = 1.0
    pred_matrix = np.zeros([64, 64, 15])
    for i in range(15):
        for n in range(64):
            for m in range(64):
                if (pred_map[n][m] == i).any():
                    pred_matrix[n, m, i] = 1.0
    iou_list = []
    length = mask.shape[0]
    wideth = mask.shape[1]
    # find the pixel that at the same time in two map
    for c in range(15):
        p_in_two = 0
        for i in range(length):
            for j in range(wideth):
                if (pred_matrix[i,j,c] != 0) and (mask_matrix[i,j,c] != 0):
                    p_in_two = p_in_two + 1
        # find the pixel that not equal to zero
        p_all = 0
        for i in range(length):
            for j in range(wideth):
                if (pred_matrix[i,j,c] != 0) or (mask_matrix[i,j,c] != 0):
                    p_all = p_all + 1
        if p_all!= 0:
            part_iou = p_in_two/p_all
            iou_list.append(part_iou)
    result = np.mean(iou_list)
    return result

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

# output: (None,4,64,64,15)
output = model.get_output()
loss = model.loss
out = output[0,7]
pred_seg = tf.argmax(out,axis=2)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# train
iteration = []
train_acc_list = []
train_iou_list = []
test_acc_list = []
test_iou_list = []
for i in range(5001):
    x_b,y_b = get_next_batch(train_data,train_label)
    x,y = get_augmentor_batch(x_b,y_b,6)
    sess.run(train_step,feed_dict={model.img:x, model.gtMaps:y})
    if i%50 == 0:
        x_for_eval = np.expand_dims(x_b,axis=0)
        pred_map = sess.run(pred_seg,feed_dict={model.img:x_for_eval})
        train_acc = cal_acc(pred_map,y_b)
        train_iou = cal_iou(pred_map,y_b)

        test_x,test_y = get_next_batch(test_data,test_label)
        test_x_for_eva = np.expand_dims(test_x,axis=0)
        pred_map = sess.run(pred_seg, feed_dict={model.img: test_x_for_eva})
        test_acc = cal_acc(pred_map, test_y)
        test_iou = cal_iou(pred_map, test_y)

        print('iter:',i,'train_acc:',train_acc,'train_iou:',train_iou,'test_acc:',test_acc,'test_iou:',test_iou )

        iteration.append(i)
        train_acc_list.append(train_acc)
        train_iou_list.append(train_iou)
        test_acc_list.append(test_acc)
        test_iou_list.append(test_iou)

# save iou and acc
iterdf = pd.DataFrame(iteration,columns=['iter'])
tradf = pd.DataFrame(train_acc_list,columns=['train_accracy'])
tridf = pd.DataFrame(train_iou_list,columns=['train_iou'])
teadf = pd.DataFrame(test_acc_list,columns=['test_accracy'])
teidf = pd.DataFrame(test_iou_list,columns=['test_iou'])
eval_info = pd.concat([iterdf,tradf,tridf,teadf,teidf],axis=1)
eval_info.to_csv('training_info_for_drawing.csv')

# save the network
path = "my_net"
is_exist = os.path.exists(path)
if not is_exist:
    os.makedirs(path)
full_path = os.path.join(path,'save_net.ckpt')
save_path = saver.save(sess,full_path)