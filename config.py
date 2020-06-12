from cx_model.resnet import resnet50
from cx_model.alexnet import alexnet
from cx_model.vgg import vgg16

save_dir = 'model_68/'

MODEL = resnet50(pretrained=True)
pthfile = ''  # /Disk1/chenxin/model/model_61/net_050.pth

Epoch = 50
BatchSize = 32

Optimizer = 'adam'
lr = 0.0001
wd = 5e-3

tensorboard_dir = '68/'

