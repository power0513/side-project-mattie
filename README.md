# side-project-mattie
# autodownloadpics
from keras.models import load_model
import cv2
import numpy as np
import os

cls_list = ['mattie', 'others']
model = load_model('face_predict_mode.h5')
predict_set = '/Users/chuang/Desktop/predict/'

for img_name in os.listdir('/Users/chuang/Desktop/predict/'):
    origin_img = cv2.imread(f'{predict_set}/{img_name}')
    img = cv2.resize(origin_img, (64, 64))
    img = np.expand_dims(img, axis=0)
    prediction_confidence = model.predict(img)
    print(prediction_confidence)
    result = cls_list[np.argmax(prediction_confidence, axis=1)[0]]
    print(result)
    cv2.imshow('img', origin_img)
    key = cv2.waitKey(0) & 0xFF   #waitkey 等待時間
    if key == ord('q'):
        break

cv2.destroyAllWindows()
