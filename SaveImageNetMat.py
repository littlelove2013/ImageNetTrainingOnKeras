from keras_preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input,MobileNet
from keras.applications.resnet50 import preprocess_input as bias_subtract_preprocess_input,ResNet50
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import keras
import scipy.io as sp
from math import *
import gc
def release(data):
    del data
    gc.collect()
# print(bias_subtract_preprocess_input)
def make_mat(df,load_path,save_path):
    samples=len(df)
    crop_size=(224,224)
    best_number=10000#每best_number个文件存成一个tfrecord
    imgs=[]
    labels=[]
    iters=0
    lens=len(df)
    file_number=0
    filenames=[]
    path=save_path+"-%d.mat"%(file_number)
#     for img_name,label in zip(df["images"],df["labels"]):
    for i in range(len(df)):    
        img_name=df["images"][i]
        label=df["labels"][i]
        img = cv2.imread(load_path+img_name)
        img = cv2.resize(img, crop_size)
        img = img.astype(np.uint8)
        imgs.append(img)
        labels.append(label)
        iters+=1
        print("\r process %s %.2f%%"%(path,iters*100.0/lens),end="")
        if iters%best_number==0:
            file={"x":np.array(imgs,np.uint8),"y":np.array(labels,np.int32)}
            sp.savemat(path,file)
            filenames.append(path)
            #清空imgs和labels
            release(imgs)
            imgs=[]
            release(labels)
            labels=[]
            a=!du -h $path
            print("\r process %s %.2f%% %s"%(path,iters*100.0/lens,a[0]))
            file_number+=1
            path=save_path+"-%d.mat"%(file_number)
    if not iters%best_number==0:#如果最后一批刚好不成整数，则单独存一份
        file={"x":np.array(imgs,np.uint8),"y":np.array(labels,np.int32)}
        sp.savemat(path,file)
        filenames.append(path)
        #清空imgs和labels
        release(imgs)
        imgs=[]
        release(labels)
        labels=[]
        a=!du -h $path
        print("\r process %s %.2f%% %s"%(path,iters*100.0/lens,a[0]))
    print("write done!")
    return filenames
val_save_path="/data/imagenet/ilsvrc12/matdata/val"
#shuffle 
val_df=val_df.sample(frac=1).reset_index(drop=True) 
print(val_df.head())
make_mat(val_df,val_path,val_save_path)
