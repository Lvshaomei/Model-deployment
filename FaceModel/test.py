import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets import SiameseNetworkDataset
# from siamese import SiameseNetwork
from siamese import Siamese

def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


testing_dir = "D:\lvshaomei\\face_race\my_project\init_data\\toUser\\train\data"
input_shape = [224, 224]

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
folder_dataset_test = dset.ImageFolder(testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transform,
                                        should_invert=False)
test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

net = Siamese(input_shape) ###
# net = SiameseNetwork(input_shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

net.load_state_dict(torch.load("D:/lvshaomei/face_race/xuxuxuxu-master/xuxuxuxu-master/Siamese_for_Face/Siamese_for_Face/weights/vgg.pkl"))

if __name__ == '__main__':

    for i in range(10):
        _, x1, label2 = next(dataiter)
        x0, x1, label2 = x0.to(device), x1.to(device), label2.to(device)
        concatenated = torch.cat((x0, x1), 0)
        # output1, output2 = net(Variable(x0), Variable(x1))
        output = net(Variable(x0), Variable(x1))[0]
        output = torch.nn.Sigmoid()(output)
        # euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated).cpu(),
               'similarity: {0:.2f}'.format(output.item()))
        # imshow(torchvision.utils.make_grid(concatenated).cpu(), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
