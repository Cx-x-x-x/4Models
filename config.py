import torch
from torch import nn

from cx_model.resnet import resnet50
from cx_model.alexnet import alexnet
from cx_model.vgg import vgg16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_dir = 'model_test/'

MODEL = resnet50(pretrained=False).to(device)
# modify
''' ResNet, Inception '''
fc_feature = MODEL.fc.in_features
MODEL.fc = nn.Linear(fc_feature, 3).to(device)
''' AlexNet, VGG '''
# cl_feature = MODEL.classifier[6].in_features
# MODEL.classifier[6] = nn.Linear(cl_feature, 3).to(device)

pthfile = '/Disk1/chenxin/model/resnet50-19c8e357.pth'  #

Epoch = 50
BatchSize = 32

Optimizer = 'adam'
lr = 0.0001
wd = 5e-3

SaveFreq = 10

tensorboard_dir = 'test/'

