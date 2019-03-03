import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 60, frames = 41):
    dict = {}
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    count = 0
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            label = fn.split('/')[2].split('-')[1]
            if label in ['0','1', '2', '3','4','5','6','7']:
                sound_clip,s = librosa.load(fn)
                for (start,end) in windows(sound_clip,window_size):
                    if(len(sound_clip[start:end]) == window_size):
                        signal = sound_clip[start:end]
                        melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                        logspec = librosa.amplitude_to_db(melspec)
                        logspec = logspec.T.flatten()[:, np.newaxis].T
                        log_specgrams.append(logspec)
                        labels.append(label)
                        dict[count] = fn;
                        count += 1

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels,dtype = np.int), dict

def one_hot_encode(labels):
    n_labels = len(labels)
    print('Label length')
    print(len(labels))
    print(labels)
    n_unique_labels = len(np.unique(labels))
    print(n_unique_labels)
    keep_indices = ['0','1', '2', '3','4','5','6','7']
    for i in range(len(labels)):
        for j in range(len(keep_indices)):
            if labels[i] == keep_indices[j]:
                labels[i] = j
    print(labels)
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    print(one_hot_encode)
    return one_hot_encode


parent_dir = '../UrbanSound8K/audio'
sub_dirs= ['fold1','fold2']#,'fold3','fold4','fold5'
    #,'fold6','fold7','fold8','fold9','fold10']
features,labels, dict = extract_features(parent_dir,sub_dirs)
labels = one_hot_encode(labels)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='SAME')



rnd_indices = np.random.rand(len(labels))

train_x = features[rnd_indices<0.5]
train_y = labels[rnd_indices<0.5]
valid_x = features[(rnd_indices>=0.5) & (rnd_indices<0.75)]
valid_y = labels[(rnd_indices>=0.5) & (rnd_indices<0.75)]
test_x = features[rnd_indices>=0.75]
test_y = labels[rnd_indices>=0.75]
print(train_x.shape)
print(train_y.shape)
print(valid_x.shape)
print(valid_y.shape)
print(test_x.shape)
print(test_y.shape)
print(rnd_indices>=0.75)

dict2 = {}
test_indices = rnd_indices>=0.75
count = 0
for i in range(len(labels)):
    if test_indices[i]:
        dict2[count] = dict[i]
        count += 1
print('count')
print(count)


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


frames = 41
bands = 60
num_channels = 2
num_labels = test_y.shape[1]

from keras import regularizers
from sklearn import metrics
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Activation, Dense, Dropout
import os

# this model implements the 5 layer CNN described in https://arxiv.org/pdf/1608.04363.pdf
# be aware, there are 2 main differences:
# the input is 60x41 data frames with 2 channels => (60,41,2) tensors
# the paper seems to report using 128x128 data frames (with no mention of channels)
# the paper also uses a receptive field size of 5x5 - as our input is smaller, I'm using 3x3

f_size = 3

model = Sequential()

model.add(Convolution2D(24, f_size, f_size, border_mode='same', input_shape=(bands, frames, num_channels)))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, f_size, f_size, border_mode='same'))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, f_size, f_size, border_mode='valid'))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(64, W_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, W_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

model.fit(train_x, train_y, validation_data=(valid_x, valid_y), callbacks=[earlystop], batch_size=32, nb_epoch=50)

y_prob = model.predict_proba(test_x, verbose=0)
print(y_prob.shape)
print(y_prob[0])
print(y_prob[1])
y_pred = y_prob.argmax(axis=-1)
print(y_pred.shape)
print(y_pred[0])
y_true = np.argmax(test_y, 1)
print(y_true.shape)
print(y_true[0])
roc = metrics.roc_auc_score(test_y, y_prob)
print ("ROC:", round(roc,3))
print(test_y.shape)
print(test_y[0])
print(test_y[1])

score, accuracy = model.evaluate(test_x, test_y, batch_size=32)
print("\nAccuracy = {:.2f}".format(accuracy))

def sanity_check(index):
    correct_count = 0
    false_positive_count = 0
    missed_count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == index or test_y[i][index] == 1:
            if y_pred[i] == index and test_y[i][index] == 1:
                correct_count +=1
            elif y_pred[i] == index:
                false_positive_count += 1
                print('false positive')
                print(i)
                print(dict2[i])
            else:
                missed_count += 1
                print('miss')
                print(i)
                print(y_pred[i])
                print(dict2[i])
    total = missed_count + false_positive_count + correct_count
    gun_accuracy = correct_count / total
    false_positive = false_positive_count / total
    missed = missed_count / total
    print("\nAccuracy = {:.2f}".format(gun_accuracy))
    print("\nPositives = {:.2f}".format(false_positive))
    print("\nReports = {:.2f}".format(missed))

sanity_check(6)

model_json = model.to_json()
with open("test_model2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model2.h5")
print("Saved model to disk")
