import pickle

import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import torch
from torch.utils.data import DataLoader

from my_datasets_process import my_datasets_process
# from siamese import SiameseNetwork
from senet_154_sia import my_net_for_faces

from my_addp_noise import AddPepperNoise
import albumentations as A
from sklearn.metrics import roc_auc_score
# from run import main
import os
import natsort
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import pickle

# my_train_path = r"E:\shao_mei_project\CeleA_face_train_and_CASIA_1_500"
my_train_path = r"D:\lvshaomei\face_race\my_project\init_data\toUser\train\CeleA_face_train_and_CASIA_1_500"

# my_train_path = r"D:\lvshaomei\face_race\my_project\init_data\toUser\train\modelTest\trainTest"


pred_dir = r'E:\shao_mei_project\my_project_Lv_shao\init_data\toUser\train\data'
# pred_dir = r"D:\lvshaomei\face_race\my_project\init_data\toUser\train\modelTest\datatest"
# r_save_path = r'E:\shao_mei_project\my_project_Lv_shao\model\result.csv'

r_save_path = r"D:\lvshaomei\face_race\my_project\result\result(1)"
# r_save_path = r"D:\lvshaomei\face_race\my_project\init_data\toUser\train\modelTest\result(1)"
# t_annos_path = r"E:\shao_mei_project\my_project_Lv_shao\init_data\toUser\train\\annos.csv"

t_annos_path = r"D:\lvshaomei\face_race\my_project\init_data\toUser\train\\annos.csv"
# t_annos_path = r"D:\lvshaomei\face_race\my_project\init_data\toUser\train\\modelTest\annosTest.csv"

train_batch_size = 16  # 16

train_number_epochs = 200 # 200

in_shape = [256, 256]
print("batch size is:",train_batch_size)
print("epochs is:",train_number_epochs)
print("sahpe is:",[256, 256])

transform = transforms.Compose([transforms.Resize((256, 256)),
                                AddPepperNoise(0.9,0.5),
                                transforms.ToTensor()])


albu_transfomem = A.Compose([A.HorizontalFlip(p = 0.5),
                             A.OneOf([
                                 A.IAAAdditiveGaussianNoise(),
                                 A.GaussNoise(var_limit=(10,80))
                             ],p=0.8),
                             A.ShiftScaleRotate(scale_limit = 0.1,rotate_limit=15,p=0.6)
                             ],p=0.6)
folder_dataset = dset.ImageFolder(root=my_train_path)

my_train_dataset = my_datasets_process(imageFolderDataset=folder_dataset,
                                        transform=[albu_transfomem,transform],
                                        should_invert=False)
print('train data size',len(my_train_dataset))
train_dataloader = DataLoader(my_train_dataset,
                              shuffle=True,
                              num_workers=0,
                              batch_size=train_batch_size)

val_dataloader = DataLoader(my_train_dataset,
                              shuffle=False,
                              num_workers=0,
                              batch_size=1)

net = my_net_for_faces(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), 0.0001, betas=(0.9, 0.999))

counter = []
loss_history = []
loss_sum = 0
train_data_len = len(my_train_dataset)
iteration_number = 0
auc = 0

def predict(net,file1,file2):
    img1 = Image.open(file1)
    img2 = Image.open(file2)
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    img1 = transform(img1)
    img2 = transform(img2)
    img1 = torch.unsqueeze(img1,dim=0)
    img2 = torch.unsqueeze(img2,dim=0)


    img1 = img1.to(device)
    img2 = img2.to(device)
    with torch.no_grad():
    # concatenated = torch.cat((img1, img2), 0)
        output = net(Variable(img1), Variable(img2))[0]
        output = torch.nn.Sigmoid()(output)
    #     print('output:',output)
    # print('finish......')
    """
        以下是进行判断的代码
        此处省略直接返回0.2
    """
    return output

def f_dir(net,to_pred_dir,result_save_path):
    subdirs = natsort.natsorted(os.listdir(to_pred_dir)) # name
    labels = []
    for subdir in tqdm(subdirs):
        result = predict(net,os.path.join(to_pred_dir,subdir,'a.jpg'),os.path.join(to_pred_dir,subdir,'b.jpg'))

        labels.append(result.item())
    fw = open(result_save_path,"w")
    fw.write("id,label\n")
    for subdir,label in zip(subdirs,labels):
        fw.write("{},{}\n".format(subdir,label))
    fw.close()

def cal_AUC(true_annos_path,pred_path):
    f_true = open(true_annos_path,'r')
    y_true = f_true.readlines()
    f_true.close()
 
    f_pred = open(pred_path,'r')
    y_pred = f_pred.readlines()
    f_pred.close()
    y_true = y_true[1:]
    y_true1 = [int(i.split(',')[1]) for i in y_true]
    y_pred = y_pred[1:]
    y_pred1 = [float(i.split(',')[1]) for i in y_pred]
    # print(y_true1)
    AUC = roc_auc_score(y_true1,y_pred1)
    return AUC

if __name__ == '__main__':
    torch.manual_seed(22)
    np.random.seed(22)
    torch.cuda.manual_seed(22)
    train_number_epochs = 200 # 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=train_number_epochs)
    for epoch in range(0, train_number_epochs):
        loss_sum = 0
        net = net.train()
        for i, data in enumerate(train_dataloader, 0):
            # if i == 5:
            #     break
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(img0, img1)
            loss_contrastive = criterion(output, label)
            loss_contrastive.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss_contrastive.item() * img1.shape[0]
            print("batch size loss is:",loss_contrastive.item())
        print("lr is: ",optimizer.param_groups[0]['lr'],'train mean loss is:',loss_sum/train_data_len)
        net.eval()
        with torch.no_grad():
            f_dir(net,pred_dir,r_save_path)
            now_auc = cal_AUC(t_annos_path,r_save_path)
            print("auc is: ",now_auc,)
            if now_auc >= auc:
                auc = now_auc
                print("--------------start---------")
                torch.save(net.state_dict(), str(epoch) +'auc_'+"{:.3%}".format(now_auc)+'.pkl')
                print("--------------torvh save---------")
                # pickle.dump(net,open("model.pkl", "wb"))###########
                # print("--------------pickle dump---------")
        torch.save(net.state_dict(), 'lastmodel'+'.pth')
        print("--------------end---------")

