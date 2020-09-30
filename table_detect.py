#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table detect with yolo 
@author: chineseocr
"""
######################################################
# 目标检测，检测表格
######################################################

import cv2
import numpy as np
from config import tableModelDetectPath
from utils import nms_box, letterbox_image, rectangle

#加载模型配置和权重
tableDetectNet = cv2.dnn.readNetFromDarknet(tableModelDetectPath.replace('.weights', '.cfg'), tableModelDetectPath)  #


def table_detect(img, sc=(416, 416), thresh=0.5, NMSthresh=0.3):
    """
    表格检测
    :param img: GBR, 要检测的图片
    :param sc: 预处理后图像的目标尺寸，一般有几个建议的值
    :param thresh: 置信度阈值，大于此置信度的才保留
    :param NMSthresh: 极大值抑制阈值
    :return:
    """
    scale = sc[0]
    #获取img的前2位，图片的高度和宽度
    img_height, img_width = img.shape[:2]
    # 输入的Blob bbox, 新的宽度和原宽度的比值, 新的高度和原高度的比值
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (scale, scale))
    # 对输入图像进行预处理，均值，缩放，通道交互[H,W,C]-->[B,C,H,W]
    inputBlob = cv2.dnn.blobFromImage(inputBlob, scalefactor=1.0, size=(scale, scale), swapRB=True, crop=False);
    #设置模型的输入
    tableDetectNet.setInput(inputBlob / 255.0)
    # 返回没有连接的输出的layer的名字，
    outputName = tableDetectNet.getUnconnectedOutLayersNames()
    # 运行前向计算，计算OutputName的layers的输出, outputs输出结果的列表
    outputs = tableDetectNet.forward(outputName)
    #存放类别id，置信度，bbox
    class_ids = []
    confidences = []
    boxes = []
    #对于多个输出结果过滤
    for output in outputs:
        #处理每个结果, detection输出格式是[centerx,centery,w,h,xxxx, class1_confidence, class2_confidence]
        # centerx 是bbox中心点坐标，w，h是bbox的宽和高
        for detection in output:
            #第5个和第6个是对每个类别的预测的置信度
            scores = detection[5:]
            #置信度最大的index是对应的是类别id
            class_id = np.argmax(scores)
            #获取对应的置信度
            confidence = scores[class_id]
            #检查置信度是否大于阈值
            if confidence > thresh:
                #还原到原图像的x，y，w，h
                center_x = int(detection[0] * scale / fx)
                center_y = int(detection[1] * scale / fy)
                width = int(detection[2] * scale / fx)
                height = int(detection[3] * scale / fy)
                #bbox左顶点（x，y），这里用left是x，top是y
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # 如果类别id是1
                if class_id == 1:
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    #计算bbox左上角和右下角的点的坐标
                    xmin, ymin, xmax, ymax = left, top, left + width, top + height
                    xmin = max(xmin, 1)
                    ymin = max(ymin, 1)
                    xmax = min(xmax, img_width - 1)
                    ymax = min(ymax, img_height - 1)
                    boxes.append([xmin, ymin, xmax, ymax])
    #bboxes的列表
    boxes = np.array(boxes)
    #对应的confidences列表
    confidences = np.array(confidences)
    #NMS非极大值抑制过滤bbox
    if len(boxes) > 0:
        boxes, confidences = nms_box(boxes, confidences, score_threshold=thresh, nms_threshold=NMSthresh)

    boxes, adBoxes = fix_table_box_for_table_line(boxes, confidences, img)
    return boxes, adBoxes, confidences


def point_in_box(p, box):
    x, y = p
    xmin, ymin, xmax, ymax = box
    if xmin <= x <= xmin and ymin <= y <= ymax:
        return True
    else:
        return False


def fix_table_box_for_table_line(boxes, confidences, img):
    """
    修正表格用于表格线检测
    :param boxes:
    :param confidences:
    :param img:
    :return: adBoxes格式 xmin, ymin, xmax, ymax
    """
    h, w = img.shape[:2]
    n = len(boxes)
    adBoxes = []
    #每个bbox进行处理
    for i in range(n):
        prob = confidences[i]

        xmin, ymin, xmax, ymax = boxes[i]
        padx = (xmax - xmin) * (1 - prob)
        padx = padx

        pady = (ymax - ymin) * (1 - prob)
        pady = pady
        xminNew = max(xmin - padx, 1)
        yminNew = max(ymin - pady, 1)
        xmaxNew = min(xmax + padx, w)
        ymaxNew = min(ymax + pady, h)

        adBoxes.append([xminNew, yminNew, xmaxNew, ymaxNew])

    return boxes, adBoxes


def crop_img(img_name, img, adBoxes):
    """
    把目标截取出来保存为图片格式
    :param img_name: 源文件的名称
    :param img: 源文件是cv2格式
    :param adBoxes: bboxing的坐标列表，可以有多个
    :return:
    """
    img_name_split = img_name.split('.')
    img_name_prefix = '.'.join(img_name_split[:-1])
    image_name_extention = '.' + img_name_split[-1]
    for idx, bbox in enumerate(adBoxes):
        xmin, ymin, xmax, ymax = [int(b) for b in bbox]
        crop_img = img[ymin:ymax,xmin:xmax]
        name = img_name_prefix + f'_cropimg_{idx}' + image_name_extention
        cv2.imwrite(name, crop_img)
        print(f'已经把表格保存到{name}')

if __name__ == '__main__':
    import time
    # p = 'img/table-detect.jpg'
    p = 'img/table_without_line.jpg'
    #读取图片
    img = cv2.imread(p)
    t = time.time()
    #获取bbox
    boxes, adBoxes, scores = table_detect(img, sc=(416, 416), thresh=0.5, NMSthresh=0.3)
    print(time.time() - t, boxes, adBoxes, scores)
    #把adBoxed画到图像上
    newimg = rectangle(img, adBoxes)
    newimg.save('img/table_without_line_detect.png')
    # img.save('img/table_detect.png')
    #截取表格图像,并保存
    crop_img(img_name=p, img=img, adBoxes=adBoxes)
