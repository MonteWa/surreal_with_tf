from __future__ import division
import tensorflow as tf
import numpy as np
from hourglass_tiny import HourglassModel
import ConfigParser
import os
import pandas as pd
from keras.preprocessing import image
import scipy.io
from skimage.transform import resize
import gc

# read the all names of files that contains person
# get the path of masks and the path of data
person_names_mat = os.listdir('VOC2010/masks')
person_names_jpg = []
for i in range(len(person_names_mat)):
    temp = person_names_mat[i][:-4]+'.jpg'
    person_names_jpg.append(temp)
mat_path = []
for i in range(len(person_names_mat)):
    path = os.path.join('VOC2010/masks',person_names_mat[i])
    mat_path.append(path)
jpg_path = []
for i in range(len(person_names_jpg)):
    path = os.path.join('VOC2010/JPEGImages',person_names_jpg[i])
    jpg_path.append(path)
data_table = pd.DataFrame(jpg_path,columns=['jpg_path'])
mat_path_df = pd.DataFrame(mat_path,columns=['mat_path'])
data_table = pd.concat([data_table,mat_path_df],axis=1)
data_table.head()

def squre_crop(img,mask):
    # overlap img and mask then crop
    overlap = np.zeros([img.shape[0],img.shape[1],img.shape[2]+1])
    overlap[:,:,0:3] = img
    overlap[:,:,3] = mask
    # random crop
    height = overlap.shape[0]
    width = overlap.shape[1]
    if height%2 != 0:
        height = height - 1
    if width%2 != 0:
        width = width - 1
    if width>height:
        range1 = [0,height]
        range2 = [width-height,width]
        range3 = [int(width/2-height/2),int(width/2+height/2)]
        crop1 = overlap[:,range1[0]:range1[1],:]
        crop2 = overlap[:,range2[0]:range2[1],:]
        crop3 = overlap[:,range3[0]:range3[1],:]
    else:
        range1 = [0,width]
        range2 = [height-width,height]
        range3 = [int(height/2-width/2),int(height/2+width/2)]
        crop1 = overlap[range1[0]:range1[1],:,:]
        crop2 = overlap[range2[0]:range2[1],:,:]
        crop3 = overlap[range3[0]:range3[1],:,:]
    return crop1,crop2,crop3

# read and preprosess the data
# read img
Data = np.zeros([3*len(data_table),256,256,4])
for i in range(len(data_table)):
    test_img = image.load_img(data_table['jpg_path'][i])
    test_img = image.img_to_array(test_img)
    # read mask
    test_mask = scipy.io.loadmat(data_table['mat_path'][i])['single_map']
    crop1,crop2,crop3 = squre_crop(test_img,test_mask)
    re_crop1 = resize(crop1,output_shape=[256,256])
    re_crop2 = resize(crop2,output_shape=[256,256])
    re_crop3 = resize(crop3,output_shape=[256,256])
    Data[i * 3 + 0] = re_crop1
    Data[i * 3 + 1] = re_crop2
    Data[i * 3 + 2] = re_crop3

# split the data into train and validation
rnd = np.arange(len(Data))
train_idx = rnd < len(Data)*0.8
valid_idx = rnd >= len(Data)*0.8
train_data = Data[train_idx]
val_data = Data[valid_idx]
# save some memory
del Data
gc.collect()

def get_next_batch(data):
    length = len(data)
    idx = np.random.randint(0,length)
    img = data[idx][:,:,0:3]
    mask = data[idx][:,:,3]
    return img,mask
def augmentor_with_keras(img,mask):
    temp = np.zeros([256,256,4])
    temp[:,:,0:3] = img
    temp[:,:,3]= mask
    after_rot = image.random_rotation(temp,rg=20,
                                      row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    after_zoom = image.random_zoom(after_rot,zoom_range=[0.7,1.3],
                                   row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    # random_shift
    return after_zoom[:,:,0:3],after_zoom[:,:,3]

def my_scale(aug_mask):
    for w in range(256):
        for h in range(256):
            if aug_mask[w,h]%1 != 0:
                aug_mask[w,h] = 0
    scale_map = np.zeros([64,64])
    for i in range(64):
        for j in range(64):
            scale_map[i,j] = aug_mask[i*4+2,j*4+1]
    return scale_map

def transfer_mask_to_gtmap(mask):
    mask = my_scale(mask)
    map = np.zeros([64,64,15])
    for i in range(15):
        for n in range(64):
            for m in range(64):
                if mask[n][m] == i:
                    map[n,m,i] = 1.0
    # transform to gtMap
    gtmap = np.zeros([8,64,64,15])
    for i in range(4):
        gtmap[i] = map
    return gtmap

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

def cal_acc(pred,mask):
    length = pred.shape[0]
    wideth = pred.shape[1]
    help_arr = np.zeros([length,wideth])
    for i in range(length):
        for j in range(wideth):
            if pred[i,j] == mask[i,j]:
                help_arr[i,j] = 1
    acc = np.mean(help_arr)
    return acc

def cal_iou(pred,mask):
    length = pred.shape[0]
    wideth = pred.shape[1]
    # find the pixel that at the same time in two map
    p_in_two = 0
    for i in range(length):
        for j in range(wideth):
            if (pred[i,j] != 0) and (mask[i,j] != 0):
                p_in_two = p_in_two + 1
    # find the pixel that not equal to zero
    p_all = 0
    for i in range(length):
        for j in range(wideth):
            if (pred[i,j] != 0) or (mask[i,j] != 0):
                p_all = p_all + 1
    if p_all != 0:
        iou = p_in_two/p_all
    else:
        iou = 0
    return iou

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

# output: (None,8,64,64,15)
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
for i in range(1000):
    x_b,y_b = get_next_batch(train_data)
    x,y = get_augmentor_batch(x_b,y_b,1)
    sess.run(train_step,feed_dict={model.img:x, model.gtMaps:y})
    if i%50 == 0:
        x_for_eval = np.expand_dims(x_b,axis=0)
        pred_map = sess.run(pred_seg,feed_dict={model.img:x_for_eval})
        y_b = my_scale(y_b)
        train_acc = cal_acc(pred_map,y_b)
        train_iou = cal_iou(pred_map,y_b)

        test_x,test_y = get_next_batch(val_data)
        test_x_for_eva = np.expand_dims(test_x,axis=0)
        test_y = my_scale(test_y)
        pred_map = sess.run(pred_seg, feed_dict={model.img: test_x_for_eva})
        test_acc = cal_acc(pred_map, test_y)
        test_iou = cal_iou(pred_map, test_y)

        print('iter:',i,'train_acc:',train_acc,'train_iou:',train_iou,'val_acc:',test_acc,'val_iou:',test_iou )
        print('training loss:',sess.run(model.loss, feed_dict={model.img:x, model.gtMaps:y}))

        iteration.append(i)
        train_acc_list.append(train_acc)
        train_iou_list.append(train_iou)
        test_acc_list.append(test_acc)
        test_iou_list.append(test_iou)

# save iou and acc
# save iou and acc
iterdf = pd.DataFrame(iteration,columns=['iter'])
tradf = pd.DataFrame(train_acc_list,columns=['train_accracy'])
tridf = pd.DataFrame(train_iou_list,columns=['train_iou'])
teadf = pd.DataFrame(test_acc_list,columns=['test_accracy'])
teidf = pd.DataFrame(test_iou_list,columns=['test_iou'])
eval_info = pd.concat([iterdf,tradf,tridf,teadf,teidf],axis=1)
eval_info.to_csv('training_info_with_pascal.csv')

# save the network
path = "my_net_with_pascal"
is_exist = os.path.exists(path)
if not is_exist:
    os.makedirs(path)
full_path = os.path.join(path,'save_net.ckpt')
save_path = saver.save(sess,full_path)