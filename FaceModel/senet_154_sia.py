import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from FaceModel.my_senet import senet154





def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width) * get_output_length(height)


class my_net_for_faces(nn.Module):
    def __init__(self,pre_train_flag = True):
        super(my_net_for_faces, self).__init__()



        my_senet154 = senet154(pretrained=pre_train_flag)

        self.senet_154_net = my_senet154

        flat_shape = 131072
        self.fc = nn.Sequential(
            nn.Linear(flat_shape, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1))

    def forward_once(self, x):
        output = self.senet_154_net(x)
        output = torch.flatten(output, 1)

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = output1 - output2
        output = self.fc(output)
        # output = nn.Sigmoid(output)
        return output
