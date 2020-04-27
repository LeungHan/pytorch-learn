# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

def _load_data_(DOWNLOAD_MNIST):
    # Mnist digits dataset
    if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,
    )

    # plot one example
    print(train_data.train_data.size())                 # (60000, 28, 28)
    print(train_data.train_labels.size())               # (60000)
    plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.train_labels[0])
    plt.show()

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # pick 2000 samples to speed up testing
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]

    return train_loader, test_x, test_y

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2                   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2)                 # choose max value in 2x2 area, output shape (16, 14, 14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,             # input height
                out_channels=32,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2                   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2)                 # choose max value in 2x2 area, output shape (32, 7, 7)
        )

        # FC
        self.out = nn.Linear(32*7*7, 10)    # fully connected layer, output 10 classes

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)

        return output                       # return x for visualization


if __name__ == "__main__":
    train_loader, test_x, test_y = _load_data_(DOWNLOAD_MNIST)                    # 加载数据

    net = CNN()                                                     # 创建卷积神经网络
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)           # 选取优化器
    loss_func = nn.CrossEntropyLoss()                               # 选中损失函数

    # Training
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):

            output = net(batch_x)                                       # 将数据传入神经网络

            loss = loss_func(output, batch_y)                           # 计算损失函数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0 :
                test_output = net(test_x)
                # torch.max(a,1):返回a中每一行中最大值的那个元素
                # troch.max()[1]:只返回最大值的每个索引
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum().item()) / float(test_y.size(0))
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

                print('Epoch: ', epoch, '| Step', step, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


    # print 10 predictions from test data
    test_output = net(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')



