import numpy as np
import os
import cv2
import dlib
from imutils import face_utils
import h5py
from scipy.misc.pilutil import imresize
folders = ['03', '04', '05', '07']


def save_images(person, word, instance_number, images, folder="words/"):
    result = np.zeros([160, 160])
    row_count = 0
    col_count = 0
    for i, image in enumerate(images):
        # print(row_count, col_count)
        result[row_count * 40: (row_count + 1) * 40, col_count * 40: (col_count + 1) * 40] = image
        col_count += 1
        if col_count == 4:
            col_count = 0
            row_count += 1
    result = imresize(result, [128, 128])
    path = folder + word + "/" + person + "_" + str(instance_number) + ".jpg"
    cv2.imwrite(path, result)


def process(folder, shape_predictor='shape_predictor_68_face_landmarks.dat'):
    for p, person in enumerate(os.listdir(folder)):
        for w, word in enumerate(os.listdir(folder+person)):
            for k,instance in enumerate(os.listdir(folder+person+'/'+word)):
                features = []
                for j,image in enumerate(os.listdir(folder+'/'+person+'/'+word+'/'+instance)):
                    path = folder+'/'+person+'/'+word+'/'+instance+'/'+image
                    # print(path)
                    if image[0] == 'c' and j < 16:
                        image = cv2.imread(path, 0)
                        detector = dlib.get_frontal_face_detector()
                        predictor = dlib.shape_predictor(shape_predictor)
                        rects = detector(image, 1)
                        for (i, rect) in enumerate(rects):
                            shape = predictor(image, rect)
                            shape = face_utils.shape_to_np(shape)
                        mouth = shape[48:68]
                        x1 = min(mouth[i][0] for i in range(0, 20)) - 5
                        y1 = min(mouth[i][1] for i in range(0, 20)) - 5
                        x2 = max(mouth[i][0] for i in range(0, 20)) + 5
                        y2 = max(mouth[i][1] for i in range(0, 20)) + 5
                        crop_image = image[y1:y2, x1:x2]
                        image = imresize(crop_image, [40, 40])
                        # cv2.imwrite(str(k)+str(j)+".jpg", image)
                        features.append(image)
                if len(features) < 16:
                    for hosa in range(16-len(features)):
                        features.append(features[-1])
                print(instance, person)
                save_images(person, word, k, features)

"""
process("Test words/Test_data6")
process("Test words/Test_data7")
process("Test words/Test_data8")
process("Test words/Test_data9")
process("Test words/Test_data10")
"""
process("Additional_data2/")
