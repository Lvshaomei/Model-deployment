"""
实现CASIA数据集的人脸检测并保存
CASIA-FaceV5 (000-099)
CASIA-FaceV5 (100-199)
CASIA-FaceV5 (200-299)
CASIA-FaceV5 (300-399)
CASIA-FaceV5 (400-499)
将这五个数据集进行人脸检测（提取人脸），并合并到一个文件夹  CASIA_face_train
"""
import sys

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import natsort
import numpy as np

from tqdm import tqdm
# 获得人脸特征向量
def obtain_face_feature(distImgPath, Mtcnn, resnet):
    aligned = []  # 初始化人脸对齐列表
    obtainImg = cv2.imread(distImgPath)  # 读取图片
    face = Mtcnn(obtainImg)  # 使用Mtcnn检测人脸，返回【人脸数组】

    if face is not None:   # 如果检测到人脸
        aligned.append(face[0])  # 将人脸加入到对齐列表中 ，并放到指定设备上
    aligned = torch.stack(aligned).to(device)
    with torch.no_grad():
        obtain_faces_emb = resnet(aligned).detach().cpu()  # 使用resnet模型获取人脸对应的特征向量
    # print("\n人脸对应的特征向量为：\n", known_faces_emb)
    return obtain_faces_emb, obtainImg


# 计算人脸特征向量间的欧氏距离，设置阈值，判断是否为同一个人脸
def match_faces(faces_emb, obtain_faces_emb, threshold):
    isExistDst = False # 初始化是否存在欧氏距离的标志为False
    distance = (obtain_faces_emb[0] - faces_emb[0]).norm().item() # 计算两张人脸的欧氏距离
    print("\n两张人脸的欧式距离为：%.2f" % distance)
    if (distance < threshold):  # 如果欧氏距离小于阈值
        isExistDist = True
    return isExistDist


if __name__ == '__main__':

    # 获取设备
    trainSetCASIA_Dir = sys.argv[1]
    CASIA_face_train_save_path = sys.argv[2]

    # trainSetCASIA_Dir = 'G:/learning_doc/faces_race/my_project/init_data/toUser/train/CASIA'
    # CASIA_face_train_save_path = 'G:/learning_doc/faces_race/train/CASIA_face_train'

    CASIA_dirs = natsort.natsorted(os.listdir(trainSetCASIA_Dir))  # 获取训练集中所有文件夹的名称，并按自然排序进行排序  CASIA-FaceV5 (000-099)  CASIA-FaceV5 (100-199)
    num_i = 0
    more_faces_number = 0
    no_face_number = 0
    for dirs_i in tqdm(range(len(CASIA_dirs))): # 遍历训练集中所有文件夹
        new_path = os.path.join(trainSetCASIA_Dir,CASIA_dirs[dirs_i])  # 获取当前文件夹的路径  G:/learning_doc/faces_race/my_project/init_data/toUser/train/CASIA/CASIA-FaceV5 (000-099)
        new_dirs = natsort.natsorted(os.listdir(new_path)) #['000','001',.....]
        for dir in new_dirs:
            img_path_s = os.path.join(new_path,dir)  # 获取当前文件夹的路径  G:/learning_doc/faces_race/my_project/init_data/toUser/train/CASIA/CASIA-FaceV5 (000-099)/000 001.....
            CASIA_save_path_dir = os.path.join(CASIA_face_train_save_path,dir) # G:/learning_doc/faces_race/my_project/init_data/toUser/train/CASIA/CASIA_face_train/001
            if os.path.exists(CASIA_save_path_dir) == False:
                os.makedirs(CASIA_save_path_dir)
            img_path_list = natsort.natsorted(os.listdir(img_path_s)) # [000_0.bmp,000_1.bmp,....]
            for img_path1 in img_path_list:
                image_path = os.path.join(img_path_s,img_path1) #  G:/learning_doc/faces_race/my_project/init_data/toUser/train/CASIA/CASIA-FaceV5 (000-099)/000/000_0.bmp
                num_i +=1
                # img = cv2.imread(image_path)
                img = Image.open(image_path) # 使用PIL库打开图片
                img = np.array(img)   # 将PIL格式的图片转换为numpy数组
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # 将RGB格式的图片转换为BGR格式，以便使用OpenCV库进行处理
                # img = np.fromfile(image_path)
                # cv2.imdecode(img,1)
                # img = cv2.imread(image_path)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                # print(device)
                # Mtcnn模型加载【设置网络参数，进行人脸检测】
                '''
                min_face_size：最小人脸尺寸，用于过滤太小的人脸，默认为20像素。
                thresholds：阈值列表，用于判断人脸是否存在，包括三个值，分别对应三个级别的网络（P-Net、R-Net和O-Net）。如果某个候选人脸框的得分大于对应的阈值，则认为该候选人脸框中存在人脸。这些阈值可以根据实际情况进行调整
                keep_all：是否保留所有检测到的人脸，默认为False，即只保留最大的人脸。
                device：设备名称，用于指定在哪个设备上运行模型。默认为cpu，也可以指定为cuda
            
                '''
                Mtcnn = MTCNN(min_face_size=50, thresholds=[0.2, 0.2, 0.7], keep_all=True, device=device)  ##可替换为其他算法即可
                pred_face1,_ = Mtcnn.detect(img)  # 对图片进行人脸检测和对齐，返回人脸框坐标和对齐后的图片
                # pre_face = pre_face[0]
                # print(len())
                if pred_face1 is not None:   # 如果检测到人脸框
                    if len(pred_face1) > 1:  # 如果检测到多张人脸框
                        print('more faces',image_path)
                        more_faces_number +=1

                    elif len(pred_face1) == 1:  # 如果只检测到一张人脸框
                        pre_face = pred_face1[0]  # 获取人脸框坐标

                        pt1_x = int(pre_face[0])  # 获取左上角坐标x
                        pt1_y = int(pre_face[1])  # 获取左上角坐标y
                        pt2_x = int(pre_face[2])  # 获取右下角坐标x
                        pt2_y = int(pre_face[3])  # 获取右下角坐标y

                        face = img.copy()[pt1_y:pt2_y,pt1_x:pt2_x,:]   # 截取人脸部分图像  使用切片操作从原始图像中裁剪出一个矩形区域，该区域的左上角坐标为(pt1_x,pt1_y)，右下角坐标为(pt2_x,pt2_y)，最后一个冒号:表示保留所有通道。
                        img = cv2.rectangle(img,(pt1_x,pt1_y),(pt2_x,pt2_y),(0,0,255),2)  # 在原图上绘制人脸框  (0,0,255)：矩形框的颜色，是一个三元组，表示BGR颜色空间中的红色。 2：矩形框的线宽，即绘制线条的像素宽度
                        # img = cv2.rectangle(img,(pt1_x,pt1_y),,(0,0,255),2)
                        name = os.path.split(image_path)[-1]  # 获取图片名称
                        cv2.imwrite(os.path.join(CASIA_save_path_dir,name),face)  # 将截取到的人脸图像保存到指定目录下
                        # cv2.imshow('img',img)
                        # cv2.imshow('face', face)
                        # cv2.waitKey(30)
                    # else:
                    #     print('no faces',image_path)
                else:   # 如果未检测到人脸框
                    no_face_number +=1
                    print('no faces',image_path)

    print('more_faces_number:',more_faces_number,'no_face_num:',no_face_number)  # 打印检测到多张人脸和未检测到人脸的数量
