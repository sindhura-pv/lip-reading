# Om
import os, cv2, numpy as np, dlib
from imutils import face_utils
from scipy.misc import imresize
import h5py


def data_builder(folder, shape_predictor='shape_predictor_68_face_landmarks.dat'):
    for speaker in os.listdir(folder):
        for word in os.listdir(folder+speaker):
            for instance in os.listdir(folder+speaker+"/"+word):
                features =[]
                for j,image in enumerate(os.listdir(folder+speaker+"/"+word+"/"+instance)):
                    path = folder + '/' + speaker + '/' + word + '/' + instance + '/' + image
                    if image[0] == 'c' and j < 16:
                        image = cv2.imread(path)
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

                        final_image = imresize(crop_image, [40, 20])
                        features.append(final_image)

                if len(features) < 16:
                    for hosa in range(16 - len(features)):
                        features.append(features[-1])
                features = np.array(features)
                # print(features.shape)
                file_name = "Test_cropped/"+"Test_"+speaker+"_"+word+"_"+instance+".h5"
                hf = h5py.File(file_name, 'w')
                hf.create_dataset('dataset', data=features, dtype="float32")
                hf.close()

                print(word, instance)


data_builder("Test_data/")
