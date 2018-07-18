import sys
sys.path.append('C:/Users/sindhurv/PycharmProjects/Flask_API_Dev/Hit Refresh')
from label_image import main_function
import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def acc_calc():
    conf_mat = np.zeros(shape=[10, 10])
    folder = "buffer/F01"
    word_acc_list = []
    total_acc = 0
    for classes in os.listdir(folder):
        word_acc = 0
        for image in os.listdir(folder+'/'+classes):

            clas, prob = main_function(folder+'/'+classes+'/'+image)
            if clas == classes:
                total_acc += 1
                word_acc += 1
            conf_mat[int(classes) - 1][int(clas) - 1] += 1
        print(classes, word_acc/10)
        word_acc_list.append(word_acc/10)
    print(total_acc/100)
    return total_acc/100, word_acc_list, conf_mat


total_acc, word_acc_list, conf_mat = acc_calc()
words = ["Begin", "Choose", "Connection", "Navigation", "Next", "Previous", "Start", "Stop", "Hello", "Web"]
df_cm = pd.DataFrame(conf_mat, index=words, columns=words)
heat = sn.heatmap(df_cm, annot=True)
plot = heat.get_figure()
plot.savefig("Models/F02/output_F05_20k_mb.png")
