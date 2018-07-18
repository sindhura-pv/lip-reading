import sys
import os, numpy as np
import h5py
from model2 import LipNet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
import datetime
from LipNet.lipnet.core.decoders import Decoder
from LipNet.lipnet.utils.spell import Spell
from LipNet.lipnet.lipreading.helpers import labels_to_text
import pandas as pd
import seaborn as sn


words = ['begin', "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "web"]
max_str_len = 14
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = CURRENT_PATH + '/results/'
LOG_DIR      = CURRENT_PATH + '/logs/'
dictionary = 'dictionary.txt'


def text_to_labels(text):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret


def get_padded_label(label):
    pad = np.ones((max_str_len-len(label))) * 26
    return np.concatenate((np.array(label), pad), axis=0)


def get_batch():
    folder = 'Data/'
    data_files = os.listdir(folder)
    # print(data_files)
    batch = []
    y = []
    input_length = []
    label_length = []
    for files in data_files:
        hf_new = h5py.File(folder+files, 'r')
        batch.append(hf_new.get('dataset'))
        speaker, word, instance = files.split("_")
        text = words[int(word)-1]
        label = get_padded_label(text_to_labels(text))
        y.append(label)
        input_length.append(len(batch[-1]) - 2)
        label_length.append(len(label))
    X_data = np.array(batch).astype(np.float32) / 255
    Y_data = np.array(y)
    input_length = np.array(input_length)
    # print(input_length)
    label_length = np.array(label_length)

    outputs = {'ctc': np.zeros([1000])}  # dummy data for dummy loss function
    inputs = {'the_input': X_data,
              'the_labels': Y_data,
              'input_length': input_length,
              'label_length': label_length
              }
    return inputs, outputs


def get_conf_mat(pred, label, mat):
    pred = text_to_labels(pred)
    label = text_to_labels(label)
    for i in range(min(len(pred), len(label))):
        mat[label[i], pred[i]] += 1
    return mat



def train(run_name, x_data, y_data):

    lipnet = LipNet(frames_n=x_data['the_input'].shape[1], img_w=x_data['the_input'].shape[2],
                     img_h=x_data['the_input'].shape[3], img_c=x_data['the_input'].shape[4],
                            absolute_max_string_len=max_str_len, output_size=27)
    lipnet.summary()

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    lipnet.model.load_weights('model_main4.h5')
    spell = Spell(path=dictionary)
    decoder = Decoder(greedy=True, beam_width=False,
                      postprocessors=[labels_to_text, spell.sentence])
    print(os.path.join(LOG_DIR, run_name))
    log = LOG_DIR + str(run_name)
    csv_log = LOG_DIR + "{}-{}.csv".format('training', run_name)
    output_log = OUTPUT_DIR + str(run_name) + "/" + "weights{epoch:02d}.h5"
    tensorboard = TensorBoard(log_dir=log)
    csv_logger = CSVLogger(csv_log, separator=',', append=True)
    checkpoint = ModelCheckpoint(output_log, monitor='val_loss',
                                 save_weights_only=True, mode='auto', period=1)

    lipnet.model.fit(x=x_data, y=y_data, batch_size=20, verbose=1,
                     epochs=20, validation_split=0.2)
    lipnet.model.save("model_main5.h5")
    """
    video = x_data['the_input'][0:1]
    y_pred = lipnet.predict(video)
    result = decoder.decode(y_pred,16)[0]
    print(result)
    """


def c_acc(res, sam):
    word_num = int((sam%100)/10)
    l = min(len(words[word_num]), len(res))
    acc=0
    for i in range(l):
        if res[i]==words[word_num][i]:
            acc+=1
    return acc


def tester(x_data, y_data):
    lipnet = LipNet(img_c=3, img_w=40, img_h=20, frames_n=16,
                    absolute_max_string_len=max_str_len, output_size=27)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-04)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights('model_main2.h5')

    spell = Spell(path=dictionary)
    decoder = Decoder(greedy=True, beam_width=100,
                      postprocessors=[labels_to_text, spell.sentence])
    acc=0
    mat = np.zeros((27,27))
    for sample in range(100):
        input_length = np.array([16])
        x = x_data[sample].reshape((1, 16, 40, 20, 3))
        y_pred = lipnet.predict(x)
        result = decoder.decode(y_pred, input_length)
        mat = get_conf_mat(result[1][0], words[int((sample % 100) / 10)], mat)
        if result[1][0] == words[int((sample%100)/10)]:
            acc += 1
        if sample%10==0:
            print(acc)
    print(acc)
    return mat


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    inputs, outputs = get_batch()
    #train(run_name, inputs, outputs)
    mat = tester(inputs['the_input'], inputs['the_labels'])
    alpha_list = list("abcdefghijklmnopqrstuvwxyz ")
    df_cm = pd.DataFrame(mat, index=alpha_list, columns=alpha_list)
    heat = sn.heatmap(df_cm)
    plot = heat.get_figure()
    plot.savefig("conf_mat_char_model.png")




