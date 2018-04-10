from __future__ import division
import tensorflow as tf
from hourglass_tiny import HourglassModel
import ConfigParser
import os
import pandas as pd
import scipy.io
import gc
from my_utils import *

print('verson:3')

# read the all names of files that contains person
# get the path of masks and the path of data
person_names_mat = os.listdir('VOC2010/all_person_masks')
person_names_jpg = []
for i in range(len(person_names_mat)):
    temp = person_names_mat[i][:-4]+'.jpg'
    person_names_jpg.append(temp)
mat_path = []
for i in range(len(person_names_mat)):
    path = os.path.join('VOC2010/all_person_masks',person_names_mat[i])
    mat_path.append(path)
jpg_path = []
for i in range(len(person_names_jpg)):
    path = os.path.join('VOC2010/JPEGImages',person_names_jpg[i])
    jpg_path.append(path)
data_table = pd.DataFrame(jpg_path,columns=['jpg_path'])
mat_path_df = pd.DataFrame(mat_path,columns=['mat_path'])
data_table = pd.concat([data_table,mat_path_df],axis=1)
print(data_table.head())

# read and preprosess the data
# read img
# len(data_table)
Data = np.zeros([3*len(data_table),256,256,4])
for i in range(len(data_table)):
    test_img = image.load_img(data_table['jpg_path'][i])
    test_img = image.img_to_array(test_img)
    # read mask
    test_mask = scipy.io.loadmat(data_table['mat_path'][i])['single_map']
    crop1,crop2,crop3 = random_put_into_256map(test_img,test_mask)
    Data[i * 3 + 0] = crop1
    Data[i * 3 + 1] = crop2
    Data[i * 3 + 2] = crop3

# split the data into train and validation
rnd = np.arange(len(Data))
train_idx = rnd < len(Data)*0.8
valid_idx = rnd >= len(Data)*0.8
train_data = Data[train_idx]
val_data = Data[valid_idx]
# save some memory
del Data
gc.collect()

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
out = output[0,3]
pred_seg = tf.argmax(out,axis=2)
train_step = tf.train.RMSPropOptimizer(1e-4).minimize(loss)
saver = tf.train.Saver()

# set up the using rate of gpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# restore old net to continue training
path = "my_net_with_pascal_Mar_22"
full_path = os.path.join(path,'save_net.ckpt')
saver.restore(sess,full_path)

# train
iteration = []
train_acc_list = []
train_iou_list = []
test_acc_list = []
test_iou_list = []
loss_list = []
for i in range(60001,90001):
    x_b,y_b = get_next_batch(train_data,batch_size=3)
    x,y = get_augmentor_batch_for_pascal(x_b,y_b,batch_size=3,aug_level=2)
    sess.run(train_step,feed_dict={model.img:x, model.gtMaps:y})
    if i%100 == 0:
        train_x,train_y = get_one_data(train_data)
        x_for_eval = np.expand_dims(train_x,axis=0)
        pred_map = sess.run(pred_seg,feed_dict={model.img:x_for_eval})
        y_b = my_scale(train_y)
        train_acc = cal_acc(pred_map,y_b)
        train_iou = cal_iou(pred_map,y_b)

        test_x,test_y = get_one_data(val_data)
        test_x_for_eva = np.expand_dims(test_x,axis=0)
        test_y = my_scale(test_y)
        pred_map = sess.run(pred_seg, feed_dict={model.img: test_x_for_eva})
        test_acc = cal_acc(pred_map, test_y)
        test_iou = cal_iou(pred_map, test_y)
        loss_for_list = sess.run(model.loss, feed_dict={model.img: x, model.gtMaps: y})
        print('iter:',i,'train_acc:',train_acc,'train_iou:',train_iou,'val_acc:',test_acc,'val_iou:',test_iou )
        print('training loss:',loss_for_list)

        iteration.append(i)
        train_acc_list.append(train_acc)
        train_iou_list.append(train_iou)
        test_acc_list.append(test_acc)
        test_iou_list.append(test_iou)
        loss_list.append(loss_for_list)


# save iou and acc
iterdf = pd.DataFrame(iteration,columns=['iter'])
tradf = pd.DataFrame(train_acc_list,columns=['train_accracy'])
tridf = pd.DataFrame(train_iou_list,columns=['train_iou'])
teadf = pd.DataFrame(test_acc_list,columns=['test_accracy'])
teidf = pd.DataFrame(test_iou_list,columns=['test_iou'])
lossf = pd.DataFrame(loss_list,columns=['loss'])
eval_info = pd.concat([iterdf,tradf,tridf,teadf,teidf,lossf],axis=1)
eval_info.to_csv('training_info_with_pascal_Mar_27.csv')

# save the network
path = "net_with_pascal_Mar_27"
is_exist = os.path.exists(path)
if not is_exist:
    os.makedirs(path)
full_path = os.path.join(path,'save_net.ckpt')
save_path = saver.save(sess,full_path)