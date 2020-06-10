import torch
import torchvision
import os
import argparse
import time
import matplotlib
from tensorboardX import SummaryWriter

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# from pre_data import MyDataset

from torch import nn
from torch import optim

from cx_model.resnet import resnet50
from cx_model.alexnet import alexnet
from cx_model.vgg import vgg16

from config import save_dir, pthfile, Epoch, BatchSize, Optimizer, lr, wd
from filter_weight_decay import group_weight


start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--outf', default='/Disk1/chenxin/model/' + save_dir,
                    help='folder to output images and model checkpoints')
args = parser.parse_args()


# data
train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(45),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243]),
                                      ])
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])

train_dataset = ImageFolder('/Disk1/chenxin/LSID3_5_1/train0', transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize,
                          shuffle=True, num_workers=20)

test_dataset = ImageFolder('/Disk1/chenxin/LSID3_5_1/test0', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BatchSize,
                         shuffle=False, num_workers=20)


# todo load the model
# model = resnet50(pretrained=True).to(device)
# model = alexnet(pretrained=True).to(device)
# model = vgg16(pretrained=True).to(device)
# 没用 model = torch.hub.load('pytorch/vision', 'googlenet', pretrained=True)
model = torchvision.models.googlenet(pretrained=True).to(device)

# modify
''' ResNet, Inception '''
fc_feature = model.fc.in_features
model.fc = nn.Linear(fc_feature, 3).to(device)
''' AlexNet, VGG '''
# cl_feature = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(cl_feature, 3).to(device)

# load pth
if pthfile is True:
    model.load_state_dict(torch.load(pthfile))
else:
    pass


# Loss and Optimizer
parameters = group_weight(model)  # avoid to decay the weight of BN
criterion = nn.CrossEntropyLoss()
if Optimizer == 'adam':
    optimizer = optim.Adam(parameters, lr=lr, weight_decay=wd)
if Optimizer == 'sgd':
    optimizer = optim.SGD(parameters, lr=lr, weight_decay=wd)

# learning rate decay
# lambda1 = lambda epoch: .1 if epoch > 39 else 1  # 10epoch后*0.1
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# train
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training !")  # 定义遍历数据集的次数
    with open("test_acc_loss.txt", "w") as f, open("train_acc_loss.txt", "w") as f1, open("log.txt", "w")as f2:
        for epoch in range(Epoch):
            print('\nEpoch: %d' % (epoch + 1))
            model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(train_loader, 0):
                # 准备数据
                length = len(train_loader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward + backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).sum()  # todo predicted.eq().cpu().sum()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                f2.write('\n')
                f2.flush()

            # 记录每一个epoch最后一个batch的训练准确率，为了与test的做对比
            f1.write('%03d %.3f%% %.03f' % (epoch+1, 100. * correct / total,  sum_loss / (i + 1)))
            f1.write('\n')
            f1.flush()

            # 每训练完一个epoch测试一下准确率
            print("Waiting Test!")
            with torch.no_grad():
                correct = 0
                total = 0
                test_loss = 0.0
                for data in test_loader:
                    model.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                acc = 100. * correct / total
                print('测试分类准确率为：%.3f%%' % (acc))

                # 保存模型 10 epoch
                if (epoch + 1) % 25 == 0:
                    print('Saving model......')
                    torch.save(model.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))

                # 每个epoch测试结果写入test_acc_loss.txt文件中
                f.write("%03d %.3f%% %.03f" % (epoch + 1, acc, test_loss / len(test_loader)))
                f.write('\n')
                f.flush()

                # 记录最佳测试分类准确率并写入best_acc.txt文件中
                if acc > best_acc:
                    f3 = open("best_acc.txt", "w")
                    f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                    f3.close()
                    best_acc = acc

        print("Training Finished, TotalEPOCH=%d" % Epoch)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
end = time.time()
print("final is in ", end-start)



