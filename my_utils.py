from __future__ import division
import numpy as np
from keras.preprocessing import image
import random
import configparser
import tensorflow as tf

Stacks = 4

def random_put_into_256map(img,mask):
    # overlap img and mask then crop
    overlap = np.zeros([img.shape[0],img.shape[1],img.shape[2]+1])
    overlap[:,:,0:3] = img
    overlap[:,:,3] = mask
    # check the shape and select crop operation
    height = overlap.shape[0]
    width = overlap.shape[1]
    # random put in
    canvas_list=[]
    for i in range(3):
        temp_canvas = np.zeros([256,256,4])
        canvas_list.append(temp_canvas)
    if height <= 256 and width <= 256:
        for i in range(3):
            height_margin = 256-height
            width_margin = 256-width
            x_start = random.randint(0,width_margin)
            y_start = random.randint(0,height_margin)
            canvas_list[i][y_start:y_start+height,x_start:x_start+width,:] = overlap
    if height > 256 and width < 256:
        y_start = random.randint(0,256-width)
        canvas_list[0][:,y_start:y_start+width,:] = overlap[0:256,:,:]
        y_start = random.randint(0,256-width)
        canvas_list[1][:,y_start:y_start+width,:] = overlap[-256:,:,:]
        if height%2 == 0:
            y_start = random.randint(0,256-width)
            canvas_list[2][:,y_start:y_start+width,:] = overlap[int(height/2-128):int(height/2+128),:,:]
        else:
            y_start = random.randint(0,256-width)
            canvas_list[2][:,y_start:y_start+width,:] = overlap[int((height-1)/2-128):int((height-1)/2+128),:,:]
    if height < 256 and width > 256:
        x_start = random.randint(0,256-height)
        canvas_list[0][x_start:x_start+height,:,:] = overlap[:,0:256,:]
        x_start = random.randint(0,256-height)
        canvas_list[1][x_start:x_start+height,:,:] = overlap[:,-256:,:]
        if width%2 == 0:
            x_start = random.randint(0,256-height)
            canvas_list[2][x_start:x_start+height,:,:] = overlap[:,int(width/2-128):int(width/2+128),:]
        else:
            x_start = random.randint(0,256-height)
            canvas_list[2][x_start:x_start+height,:,:] = overlap[:,int((width-1)/2-128):int((width-1)/2+128),:]
    if height > 256 and width > 256:
        y_start = random.randint(0,height-256)
        x_start = random.randint(0,width-256)
        canvas_list[0] = overlap[y_start:y_start+256,x_start:x_start+256,:]
        y_start = random.randint(0,height-256)
        x_start = random.randint(0,width-256)
        canvas_list[1] = overlap[y_start:y_start+256,x_start:x_start+256,:]
        y_start = random.randint(0,height-256)
        x_start = random.randint(0,width-256)
        canvas_list[2] = overlap[y_start:y_start+256,x_start:x_start+256,:]
    return canvas_list[0],canvas_list[1],canvas_list[2]

def get_next_batch(data,batch_size):
    x_batch = np.zeros([batch_size,256,256,3])
    y_batch = np.zeros([batch_size,256,256])
    length = len(data)
    for i in range(batch_size):
        idx = np.random.randint(0,length)
        x_batch[i] = data[idx][:,:,0:3]
        y_batch[i] = data[idx][:,:,3]
    return x_batch,y_batch

def get_one_data(data):
    length = len(data)
    idx = np.random.randint(0,length)
    img = data[idx][:,:,0:3]
    mask = data[idx][:,:,3]
    return img,mask

# looking for the head
def from_head_scale(data_array):
    sx = 128-43
    sy = 128-64
    for i in range(0,252):
        for j in range(0,252):
            max1 = np.max(data_array[:,:,3][i:i+5,j:j+5])
            min1 = np.min(data_array[:,:,3][i:i+5,j:j+5])
            if max1 == 1 and min1 ==1 :
                sx = i+2-43
                sy = j+2-64
                break
    if sx <= 0 and sy <= 0:
        scaling = data_array[0:128,0:128,:]
    if sx <= 0 and sy > 0 and sy < 128:
        scaling = data_array[0:128,sy:sy+128,:]
    if sx <= 0 and sy >= 128:
        scaling = data_array[128:,:128,:]
    if sx > 0 and sx < 128 and sy <=0:
        scaling = data_array[sx:sx+128,:128,:]
    if sx > 0 and sx < 128 and sy > 0 and sy <128:
        scaling = data_array[sx:sx+128,sy:sy+128,:]
    if sx > 0 and sx < 128 and sy >= 128:
        scaling = data_array[sx:sx+128,128:,:]
    if sx >= 128 and sy <=0:
        scaling = data_array[128:,:128,:]
    if sx >= 128 and sy > 0 and sy <128:
        scaling = data_array[128:,sy:sy+128,:]
    if sx >= 128 and sy >= 128:
        scaling = data_array[128:,128:,:]
    void_map = np.zeros([256,256,4])
    for i in range(128):
        for j in range(128):
            void_map[i*2:i*2+2,j*2:j*2+2,:] = scaling[i,j,:]
    return void_map

def augmentor_with_keras_for_pascal(img,mask):
    temp = np.zeros([256,256,4])
    temp[:,:,0:3] = img
    temp[:,:,3]= mask
    # rotate
    after_rot = image.random_rotation(temp,rg=20,
                                      row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    # zoom
    after_zoom = image.random_zoom(after_rot,zoom_range=[0.7,1.3],
                                   row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    return after_zoom[:,:,0:3],after_zoom[:,:,3]

def augmentor_with_keras_for_surreal(img,mask):
    temp = np.zeros([256,256,4])
    temp[:,:,0:3] = img
    temp[:,:,3]= mask
    # rotate
    after_rot = image.random_rotation(temp,rg=20,
                                      row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    # zoom
    after_zoom = image.random_zoom(after_rot,zoom_range=[0.7,1.3],
                                   row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    # scale
    rand_num = random.randint(0,100)
    if rand_num<50:
        after_scale = from_head_scale(after_zoom)
        # color jittering
        noise = np.random.randint(0,30,(256, 256,3))
        after_jittering = after_scale[:,:,0:3] + noise
        return after_jittering[:,:,0:3],after_scale[:,:,3]
    else:
        # color jittering
        noise = np.random.randint(0,30,(256, 256,3))
        after_jittering = after_zoom[:,:,0:3] + noise
        return after_jittering[:,:,0:3],after_zoom[:,:,3]

def augmentor_with_keras_for_upsamp(img,mask):
    temp = np.zeros([256, 256, 4])
    temp[:, :, 0:3] = img
    temp[:, :, 3] = mask
    # rotate
    after_rot = image.random_rotation(temp,rg=20,
                                      row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    # zoom
    after_zoom = image.random_zoom(after_rot,zoom_range=[0.7,1.3],
                                   row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')
    return after_zoom[:, :, 0:3], after_zoom[:, :, 3]

def my_scale(aug_mask):
    scale_map = np.zeros([64,64])
    for i in range(64):
        for j in range(64):
            scale_map[i,j] = aug_mask[i*4+2,j*4+1]
    return scale_map

def transfer_mask_for_upsamp(mask,batch_size):
    map = np.zeros([batch_size,256,256,15])
    for l in range(batch_size):
        for i in range(15):
            for n in range(256):
                for m in range(256):
                    if mask[l][n][m] == i:
                        map[l,n,m,i] = 1.0
    return map

def transfer_mask_to_gtmap(mask):
    mask = my_scale(mask)
    map = np.zeros([64,64,15])
    for i in range(15):
        for n in range(64):
            for m in range(64):
                if mask[n][m] == i:
                    map[n,m,i] = 1.0
    # transform to gtMap
    gtmap = np.zeros([Stacks,64,64,15])
    for i in range(4):
        gtmap[i] = map
    return gtmap

def get_augmentor_batch_surreal(x,y,batch_size,aug_level):
    x_batch = np.zeros([aug_level*batch_size,256,256,3])
    y_batch = np.zeros([aug_level*batch_size,Stacks,64,64,15])
    for i in range(batch_size):
        mask = y[i]
        gtmap = transfer_mask_to_gtmap(mask)
        x_batch[i]=x[i]
        y_batch[i]=gtmap
    for level in range(1,aug_level):
        for i in range(batch_size):
            aug_img,aug_mask = augmentor_with_keras_for_surreal(x[i],y[i])
            gtmap = transfer_mask_to_gtmap(aug_mask)
            x_batch[level*batch_size+i] = aug_img
            y_batch[level*batch_size+i] = gtmap
    return x_batch,y_batch

def get_augmentor_batch_for_pascal(x,y,batch_size,aug_level):
    x_batch = np.zeros([aug_level*batch_size,256,256,3])
    y_batch = np.zeros([aug_level*batch_size,Stacks,64,64,15])
    for i in range(batch_size):
        mask = y[i]
        gtmap = transfer_mask_to_gtmap(mask)
        x_batch[i]=x[i]
        y_batch[i]=gtmap
    for level in range(1,aug_level):
        for i in range(batch_size):
            aug_img,aug_mask = augmentor_with_keras_for_pascal(x[i],y[i])
            gtmap = transfer_mask_to_gtmap(aug_mask)
            x_batch[level*batch_size+i] = aug_img
            y_batch[level*batch_size+i] = gtmap
    return x_batch,y_batch

def get_augmentor_batch_for_upsamp(x,y,batch_size,aug_level):
    x_batch = np.zeros([aug_level*batch_size,256,256,3])
    y_batch = np.zeros([aug_level*batch_size,256,256,15])
    for i in range(batch_size):
        mask = y[i]
        mask_array = transfer_mask_for_upsamp(mask,batch_size=batch_size)
        x_batch[i]=x[i]
        y_batch[i]=mask_array
    for level in range(1,aug_level):
        for i in range(batch_size):
            aug_img,aug_mask = augmentor_with_keras_for_upsamp(x[i],y[i])
            mask_array = transfer_mask_for_upsamp(aug_mask,batch_size=batch_size)
            x_batch[level*batch_size+i] = aug_img
            y_batch[level*batch_size+i] = mask_array
    return x_batch,y_batch

def cal_acc(pred_map,mask):
    acc = 0.0
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
    if len(pixel_acc_list)>0:
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

def cal_iou_for_upsamp(pred_map,mask):
    mask_matrix = np.zeros([256, 256, 15])
    for i in range(15):
        for n in range(256):
            for m in range(256):
                if mask[n][m] == i:
                    mask_matrix[n, m, i] = 1.0
    pred_matrix = np.zeros([256, 256, 15])
    for i in range(15):
        for n in range(256):
            for m in range(256):
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
	params = {}
	config = configparser.ConfigParser()
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