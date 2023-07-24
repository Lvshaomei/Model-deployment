import os, sys
import cv2 as cv
import torchvision.transforms as transforms
from PIL import Image
import torch
from FaceModel.senet_154_sia import my_net_for_faces

import natsort
from tqdm import tqdm
import numpy as np



def change_size(image):
    image = np.array(image)
    # image = Image.fromarray(img1)  # narry --> Image
    # image = cv.imread(image2array, 1)  # 读取图片 image_name应该是变量
    img = cv.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv.threshold(img, 15, 255, cv.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv.cvtColor(binary_image, cv.COLOR_BGR2GRAY)
    # print(binary_image.shape)  # 改为单通道

    x = binary_image.shape[0]
    # print("高度x=", x)
    y = binary_image.shape[1]
    # print("宽度y=", y)
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度
    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    return pre1_picture  # 返回图片数据
def predict(file1,file2):
    input_shape = [256, 256]
    model = my_net_for_faces()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # cur_winw = os.path.abspath(__file__)
    # cur_winw = os.path.split(cur_winw)[0]
    model.load_state_dict(torch.load("D:/lvshaomei/modelDeploy/yolov5-streamlit-main/yolov5-streamlit-main/FaceModel/77auc_88.113%.pkl", map_location=torch.device('cpu')))
    img1 = change_size(file1)
    img2 = change_size(file2)
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    my_transform = transforms.Compose([transforms.Resize(input_shape),
                                    transforms.ToTensor()])
    img1 = my_transform(img1)
    img2 = my_transform(img2)

    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)

    probability = model(img1, img2)
    probability = torch.nn.Sigmoid()(probability)

    return probability


def cal_AUC(true_annos_path, pred_path):
    f_true = open(true_annos_path, 'r')
    y_true = f_true.readlines()
    f_true.close()

    f_pred = open(pred_path, 'r')
    y_pred = f_pred.readlines()
    f_pred.close()
    y_true = y_true[1:]
    y_true1 = [int(i.split(',')[1]) for i in y_true]
    y_pred = y_pred[1:]
    y_pred1 = [float(i.split(',')[1]) for i in y_pred]
    # print(y_true1)
    AUC = roc_auc_score(y_true1, y_pred1)
    return AUC

# def main(to_pred_dir,result_save_path):
#     subdirs = natsort.natsorted(os.listdir(to_pred_dir)) # name
#     labels = []
#     model.eval()
#     with torch.no_grad():
#         for subdir in tqdm(subdirs):
#             # print(os.path.join(to_pred_dir, subdir, "a.jpg"), os.path.join(to_pred_dir, subdir, "b.jpg"))
#             # name1 = subdir+'_0.bmp'
#             # name2 = subdir + '_1.bmp'
#             result = predict(os.path.join(to_pred_dir,subdir,'a.jpg'),os.path.join(to_pred_dir,subdir,'b.jpg'))
#             # print(result)
#             labels.append(result.item())
#         fw = open(result_save_path,"w")
#         fw.write("id,label\n")
#     for subdir,label in zip(subdirs,labels):
#         fw.write("{},{}\n".format(subdir,label))
#     fw.close()
#
# if __name__ == "__main__":
#     # to_pred_dir = sys.argv[1]
#     to_pred_dir = r"D:\lvshaomei\face_race\my_project\init_data\toUser\train\data"
#     result_save_path = r"D:\lvshaomei\face_race\my_project\result2"
#     print(to_pred_dir)
#     # result_save_path = sys.argv[2]
#     main(to_pred_dir, result_save_path)