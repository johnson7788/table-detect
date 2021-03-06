#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
@author: chineseocr
"""

######################################################
#  对图片的横线和竖线进行分类训练
######################################################

import sys
sys.path.append('.')
from table_line import model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from glob import glob
from image import gen

if __name__=='__main__':
    filepath = './models/table-line-fine.h5'##模型权重存放位置 
    
    checkpointer = ModelCheckpoint(filepath=filepath,monitor='loss',verbose=0,save_weights_only=True, save_best_only=True)
    # 学习率衰减
    rlu = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto', cooldown=0, min_lr=0)
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    # table line dataset label with labelme
    paths = glob('./train/dataset-line/*/*.json')

    #切分训练集和测试集
    trainP,testP = train_test_split(paths,test_size=0.1)
    #训练集个数，测试集个数
    print('total:',len(paths),'train:',len(trainP),'test:',len(testP))
    #批次大小
    batchsize=4
    #生成dataloader
    trainloader = gen(trainP,batchsize=batchsize,linetype=1)
    testloader = gen(testP,batchsize=batchsize,linetype=1)
    model.fit_generator(trainloader,
                    steps_per_epoch=max(1,len(trainP)//batchsize),
                    callbacks=[checkpointer],
                    validation_data=testloader,
                    validation_steps=max(1,len(testP)//batchsize),
                    epochs=30)
    
    
    
    