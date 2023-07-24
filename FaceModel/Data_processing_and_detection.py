# _*_ coding:utf-8 _*_
"""
实现CELEA数据集的人脸检测并保存
"""
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import natsort
from tqdm import tqdm
import cv2 as cv
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2 as cv
import shutil
import cv2
from PIL import Image
import sys

def read_CeleAData(all_img_path,save_path):
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    dir_start_idx = 500   ## dir start index
    dir_end_idx = 1000
    img_num = 5   #save images number
    dir_i = 500   ##表示从CELEA数据集中的第dir_i+1个文件夹开始取数据
    dir_list = natsort.natsorted(os.listdir(all_img_path))
    for i in tqdm(range(dir_i,len(dir_list))):
        save_dir = os.path.join(save_path,str(dir_start_idx))

        dir_full_path = os.path.join(all_img_path,dir_list[i])
        img_name_list  = natsort.natsorted(os.listdir(dir_full_path))
        if len(img_name_list) >=5 and dir_start_idx<=dir_end_idx:
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            for i in range(0,5):
                src_path = os.path.join(dir_full_path,img_name_list[i])
                shutil.copy(src_path,save_dir)  # 将文件从源路径 src_path 复制到目标路径 save_dir。
            # if dir_start_idx<=1000:
            dir_start_idx += 1
    print('-------------------')


def read_data_face_detection(identify_path,all_img_path,save_path):
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    f_txt = open(identify_path,'r')  ## Image读图片  打开文本文件 identify_path，并读取其中的每一行。每一行包含一个图片文件名和对应的标签目录。
    id_lists = f_txt.readlines()
    f_txt.close()
    faces_num = 0
    for i in tqdm(range(0,len(id_lists))):
        img_name,label_dir = id_lists[i].split(' ')
        label_dir = label_dir.replace('\n', '')
        sub_dir_path = os.path.join(save_path,label_dir)  # 根据标签目录创建一个子目录，并将该子目录的路径存储在 sub_dir_path 变量中。
        if os.path.exists(sub_dir_path) == False:
            os.makedirs(sub_dir_path)
        img_path = os.path.join(all_img_path,img_name)
        img = cv2.imread(img_path)
        # img = Image.open(img_path)
        # img = np.array(img)
        ############
        # img = Image.open(img_path,"rb")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # print(device)
        # mtcnn模型加载【设置网络参数，进行人脸检测】
        mtcnn = MTCNN(min_face_size=50, thresholds=[0.2, 0.2, 0.7], keep_all=True, device='cpu')  ##替换为其他算法即可
        pre_face1, _ = mtcnn.detect(img)
        # pre_face = pre_face[0]
        # print(len())
        if pre_face1 is not None:  # 如果检测到了人脸，则将人脸裁剪出来
            # if len(pre_face1) > 1:
            #     # print('more faces', image_path)
            #     more_faces_num += 1
                # for i in range(len(pre_face1)):
                #     pre_face = pre_face1[i]
                #     img = cv2.rectangle(img,(int(pre_face[0]),int(pre_face[1])),(int(pre_face[2]),int(pre_face[3])),(0,0,255),2)
                #     cv2.imshow('img',img)
                # cv2.waitKey(30)
            if len(pre_face1) == 1:
                faces_num+=1
                save_face_path = os.path.join(sub_dir_path,img_name)
                pre_face = pre_face1[0]

                pt1_x = int(pre_face[0])
                pt1_y = int(pre_face[1])
                pt2_x = int(pre_face[2])
                pt2_y = int(pre_face[3])
                if pt1_x<0:
                    pt1_x = 0
                if pt1_y <0:
                    pt1_y = 0
                if pt2_x<0:
                    pt2_x = 0
                if pt2_y<0:
                    pt2_y = 0
                face = img.copy()[pt1_y:pt2_y, pt1_x:pt2_x, :]
                cv.imwrite(save_face_path,face)


    print('faces numbers is:',faces_num)

if __name__ == "__main__":
    ####### 传递参数： D:/lvshaomei/face_race/my_project/init_data/toUser/train/CelebA/Anno/identity_CelebA.txt D:/lvshaomei/face_race/my_project/init_data/toUser/train/CelebA/Img/img_align_celeba/img_align_celeba D:/lvshaomei/face_race/my_project/init_data/toUser/train/CeleA_face_train

    # 实现CELEA数据集的人脸检测并保存
    # D:\lvshaomei\face_race\my_project\init_data\toUser\train\CelebA\Anno\identity_CelebA.txt
    # identify_path = r"G:/learning_doc/faces_race/CelebA/Anno/identity_CelebA.txt"
    # img_path = r"G:/learning_doc/faces_race/CelebA/Img/img_align_celeba/img_align_celeba"
    # save_path = r"G:/learning_doc/faces_race/CeleA_face_train"
    #
    identify_path = sys.argv[1]
    img_path = sys.argv[2]
    save_path = sys.argv[3]
    read_data_face_detection(identify_path,img_path,save_path)   # 人脸检测 暂时注释 避免再跑一次

    # 实现将CELEA数据集从500开始标号 并取每个文件夹的前五张图片
    # CeleA_face_train_path = identify_path
    # CeleA_and_CASIA_save_path = sys.argv[4]

    CeleA_face_train_path = r"D:/lvshaomei/face_race/my_project/init_data/toUser/train/CeleA_face_train"
    CeleA_and_CASIA_save_path = r"D:/lvshaomei/face_race/my_project/init_data/toUser/train/CeleA_face_train_and_CASIA_1_500(1)"

    # CeleA_face_train_path = r"G:/学习文件/人脸识别比赛/CeleA_face_train"
    # CeleA_and_CASIA_save_path = r"G:/学习文件/人脸识别比赛/CeleA_face_train_and_CASIA_1_500"
    read_CeleAData(CeleA_face_train_path,CeleA_and_CASIA_save_path)
    pass
