#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
image
@author: chineseocr
"""

import json
import base64
import numpy as np
import six
import cv2
from PIL import Image


def plot_lines(img, lines, linetype=2):
    """
    把线画到img上
    :param img:
    :param lines:
    :param linetype:
    :return:
    """
    tmp = np.copy(img)
    for line in lines:
        p1, p2 = line
        cv2.line(tmp, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 0), linetype, lineType=cv2.LINE_AA)

    return Image.fromarray(tmp)


def base64_to_PIL(string):
    """
    把encode成base64的图片转换回图片
    :param string: 源base64字符
    :return: 打开的图片
    """
    try:

        base64_data = base64.b64decode(string)
        buf = six.BytesIO()
        buf.write(base64_data)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    except:
        return None


def read_json(p):
    """
    读取labelme的json文件
    :param p: 一个labelme的json文件
    :return: 打开的图片，图片上的所有的线，和线对应的label
    """
    with open(p) as f:
        jsonData = json.loads(f.read())
    # 这张图的所有标注的线的信息， 例如['label': '0', 'line_color': [0, 0, 128], 'fill_color': [0, 0, 128], 'points': [[-3.0616171314629196e-17, 91.0], [1007.0, 91.0]], 'shape_type': 'line', 'flags': {}},
    shapes = jsonData.get('shapes')
    # 图片的base64数据
    imageData = jsonData.get('imageData')
    lines = []
    labels = []
    # 对每个标注都进行整理
    for shape in shapes:
        # 一个直线的2个点
        lines.append(shape['points'])
        [x0, y0], [x1, y1] = shape['points']
        # 这条线的label
        label = shape['label']
        # label==0是横线，否则是竖线
        if label == '0':
            # 如果是横线，那么y应该相等，如果不相等，相差很大，说明是斜线，当斜线超过一定角度，就变成竖线
            if abs(y1 - y0) > 500:
                label = '1'
        elif label == '1':
            if abs(x1 - x0) > 500:
                label = '0'

        labels.append(label)
    img = base64_to_PIL(imageData)
    return img, lines, labels


from numpy import cos, sin, pi


def rotate(x, y, angle, cx, cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    :param x:
    :param y:
    :param angle:
    :param cx:
    :param cy:
    :return:
    """
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def box_rotate(box, angle=0, imgH=0, imgW=0):
    """
    对坐标进行旋转 逆时针方向 0\90\180\270,
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    if angle == 90:
        x1_, y1_ = y2, imgW - x2
        x2_, y2_ = y3, imgW - x3
        x3_, y3_ = y4, imgW - x4
        x4_, y4_ = y1, imgW - x1

    elif angle == 180:
        x1_, y1_ = imgW - x3, imgH - y3
        x2_, y2_ = imgW - x4, imgH - y4
        x3_, y3_ = imgW - x1, imgH - y1
        x4_, y4_ = imgW - x2, imgH - y2

    elif angle == 270:
        x1_, y1_ = imgH - y4, x4
        x2_, y2_ = imgH - y1, x1
        x3_, y3_ = imgH - y2, x2
        x4_, y4_ = imgH - y3, x3
    else:
        x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_ = x1, y1, x2, y2, x3, y3, x4, y4

    return (x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_)


def angle_transpose(p, angle, w, h):
    """
    对线进行角度调整，点的位置坐标会改变
    :param p: 点坐标
    :param angle: 旋转角度
    :param w: 图片的宽
    :param h: 图片的高
    :return:
    """
    x, y = p
    if angle == 90:
        x, y = y, w - x
    elif angle == 180:
        x, y = w - x, h - y
    elif angle == 270:
        x, y = h - y, x
    return x, y


def img_argument(img, lines, labels, size=(512, 512)):
    """
    对图片数据进行增强，旋转图片角度
    :param img: 原始图片
    :param lines: 图片上的线
    :param labels: 所有线的label
    :param size: 暂未用到
    :return: 返回变换后的图片，线和新的labels
    """
    w, h = img.size
    # 首先微调角度
    if np.random.randint(0, 100) > 80:
        degree = np.random.uniform(-5, 5)
    else:
        degree = 0
    # degree = np.random.uniform(-5,5)
    newlines = []
    # 对线段调整同样角度
    for line in lines:
        p1, p2 = line
        p1 = rotate(p1[0], p1[1], degree, w / 2, h / 2)
        p2 = rotate(p2[0], p2[1], degree, w / 2, h / 2)
        newlines.append([p1, p2])
    # img = img.rotate(-degree,center=(w/2,h/2),resample=Image.BILINEAR,fillcolor=(128,128,128))
    # 对图片也调整同样角度
    img = img.rotate(-degree, center=(w / 2, h / 2), resample=Image.BILINEAR)
    # 对图片和线进行各种方向旋转，大角度
    angle = np.random.choice([0, 90, 180, 270], 1)[0]
    newlables = []
    for i in range(len(newlines)):
        p1, p2 = newlines[i]
        p1 = angle_transpose(p1, angle, w, h)
        p2 = angle_transpose(p2, angle, w, h)
        newlines[i] = [p1, p2]
        if angle in [90, 270]:
            if labels[i] == '0':
                newlables.append('1')
            else:
                newlables.append('0')
        else:
            newlables.append(labels[i])

    if angle == 90:
        img = img.transpose(Image.ROTATE_90)
    elif angle == 180:
        img = img.transpose(Image.ROTATE_180)
    elif angle == 270:
        img = img.transpose(Image.ROTATE_270)

    return img, newlines, newlables


def fill_lines(img, lines, linetype=2):
    """
    把lines，所有的横线或竖线，都画到img上
    :param img: 图片
    :param lines: 所有的线的坐标
    :param linetype: 线的粗细
    :return:
    """
    tmp = np.copy(img)
    for line in lines:
        p1, p2 = line
        cv2.line(tmp, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, linetype, lineType=cv2.LINE_AA)
    return tmp


def get_img_label(p, size, linetype=1):
    """
    返回图片的numpy格式，线的numpy格式，label的numpy格式
    :param p: 一个labelm的json文件
    :param size:要设置的图片的长和宽元祖，例如(640, 640)
    :param linetype:线的类型, 线的粗细
    :return:
    """
    # 获取一张图片，和所有的线，还有每个线的label
    img, lines, labels = read_json(p)
    # 对img和线都进行尺寸调整
    img, lines = img_resize(img, lines, target_size=512, max_size=1024)
    # 对图片数据增强，即变换后得到新的图像，
    img, lines, labels = img_argument(img, lines, labels, size)
    # 对图片增强，选择之类的变换
    img, lines, labels = get_random_data(img, lines, labels, size=size)

    # 全都变成numpy 格式
    lines = np.array(lines)
    labels = np.array(labels)
    # 存储全是横线的Img0，全是竖线的Img1
    labelImg0 = np.zeros(size[::-1], dtype='uint8')
    labelImg1 = np.zeros(size[::-1], dtype='uint8')

    # 把线放到Img上,ind是bool值索引
    ind = np.where(labels == '0')[0]
    labelImg0 = fill_lines(labelImg0, lines[ind], linetype=linetype)
    ind = np.where(labels == '1')[0]
    labelImg1 = fill_lines(labelImg1, lines[ind], linetype=linetype)
    #新建一个labelY，存储所有的横线和竖线
    labelY = np.zeros((size[1], size[0], 2), dtype='uint8')
    labelY[:, :, 0] = labelImg0
    labelY[:, :, 1] = labelImg1
    #转换成bool值
    labelY = labelY > 0
    return np.array(img), lines, labelY


from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0, b=1):
    """
    返回一个随机数
    :param a:
    :param b:
    :return:
    """
    return np.random.rand() * (b - a) + a


def get_random_data(image, lines, labels, size=(1024, 1024), jitter=.3, hue=.1, sat=1.5, val=1.5):
    '''
    随机预处理以进行数据增强
    :param image:
    :param lines:
    :param labels:
    :param size:
    :param jitter:
    :param hue:
    :param sat:
    :param val:
    :return:
    '''
    # 图片宽高
    iw, ih = image.size

    # 调整图片尺寸
    w, h = size
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    # scale = rand(.2, 2)
    scale = rand(0.2, 3)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 扭曲变换图像
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    N = len(lines)
    for i in range(N):
        p1, p2 = lines[i]
        p1 = p1[0] * nw / iw + dx, p1[1] * nh / ih + dy
        p2 = p2[0] * nw / iw + dx, p2[1] * nh / ih + dy
        lines[i] = [p1, p2]
    return image_data, lines, labels


def gen(paths, batchsize=2, linetype=2):
    """
    返回一个batchsize的数据的生成器
    :param paths: labelme的json文件列表
    :param batchsize: 批次大小
    :param linetype: 线的类型,粗细
    :return: 一个批次的数据
    """
    num = len(paths)
    i = 0
    while True:
        # sizes = [512,512,512,512,640,1024] ##多尺度训练
        # size = np.random.choice(sizes,1)[0]
        size = 640
        #初始化一个批次的图片
        X = np.zeros((batchsize, size, size, 3))
        Y = np.zeros((batchsize, size, size, 2))
        for j in range(batchsize):
            if i >= num:
                i = 0
                np.random.shuffle(paths)
            p = paths[i]
            i += 1

            # linetype=2
            img, lines, labelImg = get_img_label(p, size=(size, size), linetype=linetype)
            #把这张图片放到X和Y这个批次中
            X[j] = img
            Y[j] = labelImg

        yield X, Y


def img_resize(im, lines, target_size=600, max_size=1500):
    """
    更改图片尺寸
    :param im:
    :param lines:
    :param target_size:
    :param max_size:
    :return:
    """
    # 获取图片的宽和高
    w, h = im.size
    # 获取宽和高，比较大小
    im_size_min = np.min(im.size)
    im_size_max = np.max(im.size)
    # 图片缩放比例，目标尺寸除以最小长或宽
    im_scale = float(target_size) / float(im_size_min)
    # 是否target_size超过了max_size，如果超过，那么设置im_scale为max_size的
    if max_size is not None:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
    # 调整图片im的尺寸，根据im_scale
    im = im.resize((int(w * im_scale), int(h * im_scale)), Image.BICUBIC)
    # 对线的长度坐标也进行调整
    N = len(lines)
    for i in range(N):
        p1, p2 = lines[i]
        p1 = p1[0] * im_scale, p1[1] * im_scale
        p2 = p2[0] * im_scale, p2[1] * im_scale
        lines[i] = [p1, p2]
    return im, lines
