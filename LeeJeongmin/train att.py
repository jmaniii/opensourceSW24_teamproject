import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate, Activation,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization,add,Add,multiply,Lambda
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras_unet_collection import models, base, utils,losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
import pandas as pd
import numpy as np
import glob
import shutil
import cv2
from scipy import ndimage
from sklearn.model_selection import train_test_split,KFold
import matplotlib.pyplot as plt
from numba import jit

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 주어진 path에서 이미지와 mask를 불러와 형식을 맞춰주고, pixel값을 0~1 사이로 normalize 해주는 함수
# return된 이미지의 shape은 (N,512,512,3) mask는 (N,512,512,1) 이 될 것임.
# image 파일명이 환자id_HE.png, mask는 환자id_mask.png

def load_data(npy_path):
    print('-'*30)
    print('load images...')
    print('-'*30)
    
    for i, path in enumerate(npy_path):
#         print(path)
        mask_npy_path = path.replace('_HE','_mask')
        train_npy_path = path
    
        imgs_tmp = [cv2.imread(train_npy_path)]
        imgs_mask_tmp = [cv2.imread(mask_npy_path,0)]
#         print(imgs_tmp.shape, imgs_mask_tmp.shape)
        
        if i==0:
            imgs = imgs_tmp
            imgs_mask = imgs_mask_tmp
            
        else:
            imgs = np.append(imgs, imgs_tmp,axis=0)
            imgs_mask = np.append(imgs_mask, imgs_mask_tmp,axis=0)
    imgs_tmp,imgs_mask_tmp = 0,0
    print('-'*30)
    print('imgs : {} \nmasks : {}'.format(imgs.shape, imgs_mask.shape))    
    print('-'*30)
    imgs = imgs.astype('float32')
    imgs_mask = imgs_mask.astype('float32')
    print('img : ', imgs.max())
    print('mask : ',imgs_mask.max())

    print('-'*30)
    print('normalization start...')
    print('-'*30)
    
    imgs = cv2.normalize(imgs, None, 0, 1, cv2.NORM_MINMAX)

    imgs_mask[imgs_mask<= 127] = 0
    imgs_mask[imgs_mask > 127] = 1

    print('img : ',imgs.max())
    print('mask : ',imgs_mask.max())

    return imgs, imgs_mask
#지정된 경로의 folder가 없으면 생성해주는 함수.

def mkfolder(folder):
    if not os.path.lexists(folder):
        os.makedirs(folder)
#모델 metric으로 사용될 recall을 정의해주는 함수

def recall(y_true, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_true_yn = K.round(K.clip(y_true, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_true_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_true_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall
img_path = '../../1. 데이터/4. 골반 분할 연구 데이터/'
all_patient = np.array([i.replace('_HE.png', '') for i in os.listdir(img_path) if '_HE.png' in i])
all_patient
#5-fold cross validation을 하기 위해서 불러온 환자리스트를 나누어줌

kf = KFold(n_splits=5, random_state=5, shuffle=True)
fold = []
for train_index, test_index in kf.split(all_patient):
    train_patient, test_patient = all_patient[train_index.astype(int)], all_patient[test_index.astype(int)]    
    
    testset = [img_path+'{}_HE.png'.format(p) for p in test_patient]
    trainset = [img_path+'{}_HE.png'.format(p) for p in train_patient]

    fold.append([trainset, testset])
fold
#data 증강을 위해서 이미지 변형의 범위를 지정해주는 부분
#rotation_range 범위 내의 각도만큼 돌아가는 이미지를 생성하고
#width/height shift만큼 위치가 이동된 이미지를 생성하고
#zoom_range 범위만큼 확대/축소된 이미지를 생성하고
#horizontal_flip이 있으니까 좌우 반전된 이미지도 생성함

data_gen_args = dict(rotation_range=10.,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2, horizontal_flip=True)
for foldnum in range(0,5):
    trainset = fold[foldnum][0]
    testset = fold[foldnum][1]
    
    #model 저장 위치 생성
    #경로는 마음대로
    #fold{} 하고 foldnum은 유지 안하면 폴드별 결과물이 다 겹쳐져버리니 주의
    
    sv_model_folder ='6_result/exp_fold{}/model/'.format(foldnum)
    mkfolder(sv_model_folder)

    #Train 데이터 불러오기

    imgs_train, imgs_mask_train = load_data(trainset)
    print('='*30)
    print('-'*30)
    print("load unet model")
    print('-'*30)

    imgs_mask_train = np.expand_dims(imgs_mask_train,axis=-1)
    imgs_mask_train.shape

    imgs_train,imgs_val,imgs_mask_train,imgs_mask_val = train_test_split(imgs_train,imgs_mask_train,test_size=0.2,random_state=7)

    #data 증강을 위한 generator 선언
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    val_image_datagen = ImageDataGenerator()
    val_mask_datagen = ImageDataGenerator()

    image_generator = image_datagen.flow(imgs_train,batch_size=1,seed=1)
    mask_generator = mask_datagen.flow(imgs_mask_train,batch_size=1,seed=1)

    valt_generator = val_image_datagen.flow(imgs_val,batch_size=1,seed=1)
    valm_generator = val_mask_datagen.flow(imgs_mask_val,batch_size=1,seed=1)

    train_generator = zip(image_generator, mask_generator)
    validation_generator = zip(valt_generator,valm_generator)
    
    
    img_rows, img_cols=512,512


    #주석처리한건 swin_unet의 코드
    #적용한건 att_unet_2d 코드
    
    #kers_unet_collection을 이용해서 모델을 쉽게 구성하고 불러올 수 있음.
    #다른 모델은 https://github.com/yingkaisha/keras-unet-collection에서 참조
    
#     model = models.swin_unet_2d((512, 512, 3), filter_num_begin=64, n_labels=1, depth=4, stack_num_down=2, stack_num_up=2, 
#                             patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
#                             output_activation='Softmax', shift_window=True, name='swin_unet')

    model = models.att_unet_2d((img_rows, img_cols,3), filter_num=[64, 128, 256, 512], n_labels=1, 
                               stack_num_down=2, stack_num_up=2, activation='ReLU', 
                               atten_activation='ReLU', attention='add', output_activation='Sigmoid', 
                               batch_norm=True, pool=False, unpool=False, 
                               backbone='ResNet101V2', weights='imagenet', 
                               freeze_backbone=True, freeze_batch_norm=True, 
                               name='attunet')
    #     model = multi_gpu_model(model,gpus=4)

    learning_rate = 0.0001

    # model.compile(optimizer=Adam(learning_rate=learning_rate), 
    #               loss=[losses.dice,losses.dice,losses.dice,losses.dice,losses.dice,losses.dice],
    #               loss_weights=[0.25,0.25,0.25,0.25,1,0],
    #               metrics=['acc', 'binary_crossentropy', recall])

    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=[losses.dice],
                  metrics=['acc', 'binary_crossentropy', recall])

    # reference : https://stackoverflow.com/questions/43782409/how-to-use-modelcheckpoint-with-custom-metrics-in-keras
    # print(model.metrics_names)

    # sv_model_folder ='../4_result/exp3_RGBE/model/fold{}/'.format(num)
    # mkfolder(sv_model_folder)

    # sv_pred_folder = '../4_result/exp3_HE/pred/fold{}/'.format(num)
    # mkfolder(sv_pred_folder)

    save_check_folder = sv_model_folder+'hdf5/'
    mkfolder(save_check_folder)

    def sch(epoch):
        if epoch>30:
            return 0.001
        else:
            return 0.01

    epochs = 100
    batch_size = 1

    model_checkpoint = ModelCheckpoint(sv_model_folder+'best.h5', monitor='val_loss', verbose=1, save_best_only=True)
    sc = LearningRateScheduler(sch)
    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1, verbose=1)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    print('-'*30)
    print('start training')
    print('-'*30)

    # imgs_train = np.rollaxis(imgs_train, 3, 1)
    # imgs_mask_train = np.rollaxis(imgs_mask_train, 3, 1)
    # imgs_validation = np.rollaxis(imgs_validation, 3, 1)
    # imgs_mask_validation = np.rollaxis(imgs_mask_validation, 3, 1)
    
    #steps_per_epoch를 3008로 준 이유는 train data 752장을 4배수 증강시켜서 사용했기 때문.
    #batchsize는 1

    model.fit(train_generator, steps_per_epoch=3008,epochs=epochs, verbose=1, validation_data=validation_generator,
              validation_steps = 188,shuffle=True, callbacks=[model_checkpoint,reduceLROnPlateau,earlystopping])

    print('save model')

    model.save(sv_model_folder+'last.h5'.format(learning_rate, epochs))

    # print('predict test data')
    # imgs_mask_test = model.predict(imgs_test, batch_size=4, verbose=1)

    # pred_file_name = sv_pred_folder +'exp1.npy'
    # np.save(pred_file_name, imgs_mask_test)
    # #     num += 1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score

from tensorflow.keras.models import load_model
# 각 fold별로 생성된 모델의 성능을 측정하는 부분

for foldnum in range(5):
    
    testset = fold[foldnum][1]
    imgs_test, imgs_mask_test = load_data(testset)
    model = load_model('6_result/exp_fold{}/model/best.h5'.format(foldnum), custom_objects={"dice": losses.dice, 'recall':recall})
    print(model.summary())
    # imgs_test, imgs_mask_test = load_data(testset[:4])
    mask_pred = model.predict(imgs_test, batch_size=4, verbose=1)

    print(testset[0])
    # true_list=np.load(testset[0])
    true_list = imgs_mask_test
    true_list=true_list.astype('float32')
    # true_list = true_list/255.0
    # true_list[true_list > 0.5] = 1
    # true_list[true_list <= 0.5] = 0
    print(true_list.shape)

    pred_list=mask_pred
    # pred_list=imgs_mask_test
    pred_list[pred_list > 0.5] = 1
    pred_list[pred_list <= 0.5] = 0
    print(pred_list.shape)

    # for i in range(pred_list.shape[0]):
    #     pred = pred_list[i].astype('uint8')
    #     pred[pred <= 0.5] = 0
    #     pred[pred > 0.5] = 255
    # #     pred = fill_hole_cv(pred)
    #     pred_list[i]=pred

    # pred_list[pred_list > 127] = 1
    # pred_list[pred_list <= 127] = 0

    sensitivity=[]
    specificity=[]
    acc=[]
    dsc=[]

    for i in range(len(true_list)):
        yt=true_list[i].flatten()
        yp=pred_list[i].flatten()
        mat=confusion_matrix(yt,yp)
        if len(mat) == 2:
            ac=(mat[1,1]+mat[0,0])/(mat[1,0]+mat[1,1]+mat[0,1]+mat[0,0])
            st=mat[1,1]/(mat[1,0]+mat[1,1])
            sp=mat[0,0]/(mat[0,1]+mat[0,0])
            if mat[1,0]+mat[1,1] == 0:
                specificity.append(sp)
                acc.append(ac)
            else:
                sensitivity.append(st)  
                specificity.append(sp)
                acc.append(ac)
        else:
            specificity.append(1)
            acc.append(1)

    for i in range(len(true_list)):
        yt=true_list[i]
        yp=pred_list[i]
        if np.sum(yt) != 0 and np.sum(yp) != 0:
            dice = np.sum(yp[yt==1])*2.0 / (np.sum(yt) + np.sum(yp))
            dsc.append(dice)

    print("complete")      
    print("acc avg : {0:0.4f}".format(np.mean(acc)))
    print("sensitivity avg : {0:0.4f}".format(np.mean(sensitivity)))
    print("specificity avg : {0:0.4f}".format(np.mean(specificity)))
    print("dsc avg : {0:0.4f}".format(np.mean(dsc)))
    print('-'*30)
    print("sensitivity min:",np.min(sensitivity))
    print("specificity min:",np.min(specificity))
    print("dsc min:",np.min(dsc))
    print("acc min:",np.min(acc))
    print('-'*30)
    print("sensitivity max:",np.max(sensitivity))
    print("specificity max:",np.max(specificity))
    print("dsc max:",np.max(dsc))
    print("acc max:",np.max(acc))
    
    imgs_test, imgs_mask_test = 0,0
