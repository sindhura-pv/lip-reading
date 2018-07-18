import os
from shutil import move

s = 'C:/Users/sindhurv/Desktop/Fresh_reader/words/'
d = 'C:/Users/sindhurv/Desktop/Fresh_reader/buffer/F01/'
folders =['01','02','03', '04', '05','06','07','08','09', '10']
for i, words in enumerate(os.listdir(d)):
    for images in os.listdir(d+words):
        dest_path = s+words
        if images.split('_')[0] == 'F05':
            print(d+words+'/'+images, dest_path)
            move(d+words+'/'+images, dest_path)




