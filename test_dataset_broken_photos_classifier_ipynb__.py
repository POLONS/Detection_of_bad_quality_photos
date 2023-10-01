# -*- coding: utf-8 -*-
"""test_dataset_broken_photos_classifier.ipynb""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

#Применение на всех 258 фото - ВСПЫШКА / GLARE DETECTION жжжжж

lstclassification1 = []
stclass = []
lstpredictions = []
from PIL import ImageFile
import pandas as pd

glareCNN = tf.keras.models.load_model(path1+'/glare_detect.h5')
def glare(img_path,img_name):

    test_image1 = tf.keras.preprocessing.image.load_img(str(img_path)+str(img_name),target_size = (64,64))
    test_image2 = tf.keras.preprocessing.image.img_to_array(test_image1)
    test_image2 = np.expand_dims(test_image2, axis = 0)
    result = glareCNN.predict(test_image2/255)
    return test_image1,result

def predd(score):
    pred_class = ""
    if score[0][0] <0.04: #lowered to 4% as animals during night also displayed as white spot and this distorts analysis
        pred_class = "1"
    else:
        pred_class = "0"
    return pred_class


grade = {'Вспышка':[], 'Оценка': [], 'Наименование': []}

for i in range(0,(len(os.listdir(path_train)))):
    df2,df1 = glare(path_train+'/',os.listdir(path_train)[i])
    lstclassification = predd(df1)
    grade['Вспышка'].append(lstclassification)
    grade['Оценка'].append(df1)
    grade['Наименование'].append(str(os.listdir(path_train)[i]).lower())
    df_glare = pd.DataFrame(grade, columns = ['Вспышка','Оценка','Наименование'])

df_glare = df_glare.rename(columns={'Вспышка': 'glare', 'Оценка': 'Score_glare', 'Наименование': 'filename'})
df_glare
#eg. photos 10,11,12 were ccorrectly identified

#Применение на всех 258 фото - РАЗМЫТИЕ / BLUR  DETECTION жжжжж


total_2 = {'Размытие':[], 'Оценка': [], 'Наименование': []}
trained_model = load_model(path1+'/blur_detect.h5')
from PIL import ImageFile
import pandas as pd
pred = []
pred2=[]
score = []

source_dir = path_train
for i in range(0,(len(os.listdir(path_train)))):
    img = tf.keras.preprocessing.image.load_img(source_dir+'/'+os.listdir(path_train)[i],target_size = (128,128))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = trained_model.predict(x/255) #255 RGB
    classes2=np.array(classes)
    pred.append(img)
    score.append(classes)
    if classes[0][0]<0.5:
        pred1="1"
        total_2['Размытие'].append(pred1)
        total_2['Оценка'].append(np.squeeze(classes2,axis=1))
        total_2['Наименование'].append(str(os.listdir(path_train)[i]).lower())
        k+=1
    else:
        pred1="0"
        total_2['Размытие'].append(pred1)
        total_2['Оценка'].append(np.squeeze(classes2,axis=1))
        total_2['Наименование'].append(str(os.listdir(path_train)[i]).lower())


df_blur = pd.DataFrame(total_2, columns = ['Размытие','Оценка','Наименование'])

#Размытые определены правильно, например фото 37,38,43-54
#Некоторые из  размеченных как "размытые"  в тренировочном датасете- определены как чистые,что подтверждает, то
#что предварительный анализ может  быть с ошибками (размеченный вручную)
df_blur = df_blur.rename(columns={'Размытие': 'blur', 'Оценка': 'Score_blur','Наименование': 'filename'})
df_blur

img_path = path_train +'/' #e.g. 1
img = str(input('Введите_наименование:')).upper() +'.JPG'
image_calc,result = glare(img_path,img)
classification = predd(result)
image_calc

#Применение на всех 258 фото - ТЕМНО / DARK PHOTO DETECTION

classification_dark=''
from PIL import ImageFile
import pandas as pd
import cv2
import sys
import numpy as np
directory = path_train
i=0
grade3 = {'Темно':[], 'Оценка': [], 'Наименование': []}
for pic in os.listdir(directory):

    im = cv2.imread(path_train+'/'+pic, cv2.IMREAD_GRAYSCALE)

    meanpercent = np.mean(im) * 100 / 255  # mean brightness
    if meanpercent < 21: # needed to be increased to 21% after comparing results to actual dataset
      classification_dark = "1"
      grade3['Темно'].append(classification_dark)
      grade3['Оценка'].append(meanpercent)
      grade3['Наименование'].append(str(os.listdir(path_train)[i]).lower())
      i+=1
    else:
      classification_dark = "0"
      grade3['Темно'].append(classification_dark)
      grade3['Оценка'].append(meanpercent)
      grade3['Наименование'].append(str(os.listdir(path_train)[i]).lower())
      i+=1

df_dark = pd.DataFrame(grade3, columns = ['Темно','Оценка','Наименование'])

df_dark= df_dark.rename(columns={'Оценка': 'Score_dark','Темно': 'dark','Наименование': 'filename'})
df_dark

df_final = df_blur.copy() #save copy

#merge
df_final=df_final.merge(df_dark,right_on='filename', left_on='filename')
column_to_add=df_final["filename"]
df_final

#moving column
column_to_move = df_final.pop("filename")
df_final

df_final.insert(4, 'filename',column_to_add)

df_final

dt= df_final.copy()

#merge all
dt=dt.merge(df_glare,right_on='filename',left_on='filename')

dt

# add broken photo column, and set 1 provided that one of three niose is also positive (1)
import numpy as np
dt['broken'] = np.where((dt['blur'] == '1') | (dt['glare'] == '1') | (dt['dark'] == '1'), 1, 0)
dt
