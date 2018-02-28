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

def next_batch(data,label,batch_size):
    batch_x = np.zeros([batch_size,256,256,3])
    batch_y = np.zeros([batch_size,4,64,64,15])
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    for i in range(batch_size):
        batch_x[i] = data[idx[i]]
        batch_y[i] = label[idx[i]]
    return batch_x,batch_y