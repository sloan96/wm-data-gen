# -*- coding: utf-8 -*-
# @Time : 2019/1/29 0029 8:08
# @Author : sloan
# @File : gen_num.py
# @Software: PyCharm
import pygame
import os
import cv2
import numpy as np

path = "./ttf_test/"
img_path = "./train_pix/"
pygame.init()
#160*120,w*h
dst_w,dst_h = 160,120
def get_img(x, font_style, img_w, img_h, font_size):

    font = pygame.font.Font(path + font_style, font_size)
    rtext = font.render(x, True, (0, 0, 0), (255, 255, 255))
    w,h= font.size(x)
    print("font:",w,h)
    # assert(font_size <= img_h)
    rtext = pygame.transform.scale(rtext, (w, h))
    return rtext, w, h

def pg2cv(img_in, img_w, img_h, pad_up=0, pad_down=0, pad_left=0, pad_right=0):
    buff = pygame.image.tostring(img_in, 'RGB')
    np_s = np.fromstring(buff, np.uint8).reshape(img_h, img_w, 3)
    img = cv2.cvtColor(np_s, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(img, pad_up, pad_down, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img

def gen_trainset():

    ttf_list=[]
    dx =48
    num = 1
    for a, b, c in os.walk('ttf_test'):
        ttf_list.append(c)

    for font_name in ttf_list[0]:
        print(font_name)
        i = 0
        for fn in os.listdir(img_path):
            label = fn.split('_')[0]
            train_img_name = "img_"+str(i)+".png"
            src_img = cv2.imread(os.path.join(img_path,fn))
            train_label_name = "label_"+str(i)+".png"
            print(train_img_name,train_label_name)
            pg_img, w, h = get_img(label, font_name, len(label) * dx, dx, dx)
            img_cv2 = pg2cv(pg_img, w, h)
            img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            sz = cv2.findNonZero(255 - img_gray)
            try:
                rect = cv2.boundingRect(sz)
                xmin, ymin, rect_w, rect_h = rect
                x1, y1, x3, y3 = xmin, ymin, xmin + rect_w, ymin + rect_h
                img_gray_new = img_gray[y1:y3, x1:x3]
            except Exception as e:
                print(e)
                continue
            print("img_gray_new",img_gray_new.shape)
            img_gray_new_h,img_gray_new_w = img_gray_new.shape[:2]
            margin_top = (dst_h-img_gray_new_h)//2
            margin_left = (dst_w-img_gray_new_w)//2
            save_img = cv2.copyMakeBorder(img_gray_new,margin_top, dst_h-img_gray_new_h-margin_top, margin_left, dst_w-img_gray_new_w-margin_left, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # save_img = cv2.cvtColor(cv2.resize(save_img, (512, 512)),cv2.COLOR_GRAY2BGR)
            save_img = cv2.cvtColor(save_img, cv2.COLOR_GRAY2BGR)
            print("save_img:",save_img.shape)
            print("src:",src_img.shape)
            # shape_dst = np.min(src_img.shape[:2])
            # oh = (src_img.shape[0] - shape_dst) // 2
            # ow = (src_img.shape[1] - shape_dst) // 2
            #
            # img = src_img[oh:oh + shape_dst, ow:ow + shape_dst]
            #resize_img = cv2.resize(src_img, (512, 512))
            #print("resize_src_img:",resize_img.shape)

            cv2.imwrite(os.path.join("./train/train_B",train_img_name),src_img)
            cv2.imwrite(os.path.join("./train/train_A", train_label_name), save_img)

            i += 1
            # cv2.imshow('src',src_img)
            # cv2.imshow('resize_src,resize_img)
            # cv2.imshow('save_image', save_img)
            # cv2.imshow('gray_image', img_gray)
            # cv2.imshow('cut_image', img_gray_new)
            # cv2.waitKey()
def gen_testset():
    string = ['08950','01234','98760','78317','13145','26789','34256','46093','51980','62048','80237']

    ttf_list=[]
    dx =48
    num = 1
    for a, b, c in os.walk('ttf_test'):
        ttf_list.append(c)

    for font_name in ttf_list[0]:
        print(font_name)
        for i in range(len(string)):


            train_label_name = "label_"+str(i)+".png"

            pg_img, w, h = get_img(string[i], font_name, len(string[i]) * dx, dx, dx)
            img_cv2 = pg2cv(pg_img, w, h)
            img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            sz = cv2.findNonZero(255 - img_gray)
            try:
                rect = cv2.boundingRect(sz)
                xmin, ymin, rect_w, rect_h = rect
                x1, y1, x3, y3 = xmin, ymin, xmin + rect_w, ymin + rect_h
                img_gray_new = img_gray[y1:y3, x1:x3]
            except Exception as e:
                print(e)
                continue
            print("img_gray_new",img_gray_new.shape)
            img_gray_new_h,img_gray_new_w = img_gray_new.shape[:2]
            margin_top = (dst_h-img_gray_new_h)//2
            margin_left = (dst_w-img_gray_new_w)//2
            save_img = cv2.copyMakeBorder(img_gray_new,margin_top, dst_h-img_gray_new_h-margin_top, margin_left, dst_w-img_gray_new_w-margin_left, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            save_img = cv2.cvtColor(cv2.resize(save_img, (512, 512)),cv2.COLOR_GRAY2BGR)
            # save_img = cv2.cvtColor(save_img,cv2.COLOR_GRAY2BGR)
            print("save_img:",save_img.shape)

            # shape_dst = np.min(src_img.shape[:2])
            # oh = (src_img.shape[0] - shape_dst) // 2
            # ow = (src_img.shape[1] - shape_dst) // 2
            #
            # img = src_img[oh:oh + shape_dst, ow:ow + shape_dst]

            cv2.imwrite(os.path.join("./test/test_A", train_label_name), save_img)
            i += 1
            # cv2.imshow('src',src_img)
            # cv2.imshow('resize_src,resize_img)
            # cv2.imshow('save_image', save_img)
            # cv2.imshow('gray_image', img_gray)
            # cv2.imshow('cut_image', img_gray_new)
            # cv2.waitKey()

if __name__=="__main__":
    gen_trainset()
    #gen_testset()