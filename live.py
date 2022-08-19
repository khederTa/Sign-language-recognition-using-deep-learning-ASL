import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# label
dt = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E',
      '5':'F', '6':'G', '7':'H', '8':'I', '9':'J', '10':'K', '11':'L', '12':'M', 
      '13':'N',  '14':'O', '15':'P', '16':'Q','17':'R', '18':'S',
      '19':'T', '20':'U', '21':'V', '22':'W', '23':'X', '24':'Y', '25':'Z','26':'del','27':'space'}


sarah = 0
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
model = keras.models.load_model("model_sign_language.h5")
print(model.summary())

# preprocessing 
def prepare_image(frame):
    #frame = cv2.flip(frame, 1)
    img = cv2.resize(frame, (224, 224))
    cv2.imshow("resize", img)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

import cv2
import mediapipe as mp
from time import sleep
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

while cap.isOpened():
    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    # finalimg = prepare_image(frame)
        
    # print(finalimg.shape)
    # #cv2.imshow("crop", crop)
    # predictions = model.predict(finalimg)
    # predictions = np.rint(predictions)
    # i = -1
    # j = -1 
    # ii = 0
    # jj = 0
    # for res in predictions:
    #     i = i+1
    #     for rr in res:
    #         j = j+1
    #         #print(rr)
    #         if rr == 1:
    #             ii = i
    #             jj = j
            
    # print(dt.get(str(jj)))
    # print(ii,jj)
    #sleep(1)
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            
            wlength = x_max-x_min
            hlength = y_max-y_min
            # if wlength > hlength:
            #     dist = wlength - hlength
            #     if x_max+50+dist/2 < w:
            if wlength > hlength:
                dist = wlength - hlength
                if y_max+int(dist/2) < h and y_min-(int(dist/2)) >= 0 :
                    y_max += int(dist/2)
                    y_min -= int(dist/2)
                elif y_max+int(dist/2) < h and y_min-int(dist/2) <= 0 :
                    temp = y_min
                    y_min = 0
                    y_max = y_max + int(dist/2) + (int(dist/2) - temp)
                elif y_max+int(dist/2) > h and y_min-int(dist/2) >= 0 :
                    temp = h-y_max
                    y_max = h
                    y_min = y_min - int(dist/2) - (int(dist/2) - temp)

            else:
                dist =  hlength - wlength
                if x_max+int(dist/2) < w and x_min-int(dist/2) >= 0 :
                    x_max += int(dist/2)
                    x_min -= int(dist/2)
                elif x_max+int(dist/2) < w and x_min-int(dist/2) < 0 :
                    temp = x_min
                    x_min = 0
                    x_max = x_max + int(dist/2) + (int(dist/2) - temp)
                elif x_max+int(dist/2) > w and x_min-int(dist/2) >= 0 :
                    temp = w-x_max
                    x_max = w
                    x_min = x_min - int(dist/2) - (int(dist/2) - temp)

            if x_min-50<0:
                x_min = 0
            else:
                x_min = x_min - 50
            if y_min-50<0:
                y_min = 0
            else:
                y_min = y_min - 50 
            if x_max+50>w:
                x_max = w
            else:
                x_max = x_max + 50
            if y_max+50>h:
                y_max = h
            else:
                y_max = y_max + 50
        
            # print(x_min)
            # print(y_min)
            # print(y_max)
            # print(x_max)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            crop = frame[y_min:y_max, x_min:x_max]
            
            # print(x_max-x_min)
            # print(hlength)
            #mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
            finalimg = prepare_image(crop)
            #print(finalimg)
            #print(finalimg.shape)
            cv2.imshow("crop", crop)
            predictions = model.predict(finalimg)
            # rint this mean round int 
            
            #print("kept" ,  predictions.kept)
            #print("isDisposedInternal" ,  predictions.isDisposedInternal)
            # print("shape" ,  finalimg.shape)
            # print("dtype" ,  finalimg.dtype)
            # print("size" ,  finalimg.size)
            # print("strides" ,  finalimg.strides)
            #print("dataId" ,  predictions.dataId)
            #print("id" ,  predictions.id)
            #print("rankType" ,  predictions.rankType)
            #print("scopeId" ,  predictions.scopeId)
            predictions = np.rint(predictions)
            
            i = -1
            j = -1 
            ii = 0
            jj = 0
            for res in predictions:
                i = i+1
                for rr in res:
                    j = j+1
                    #print(rr)
                    if rr == 1:
                        ii = i
                        jj = j

            # print actual prediction (get the value from labels)  
            print(dt.get(str(jj)))
            #print(ii,jj)
            sleep(0.5)
    #press 'q' to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)
