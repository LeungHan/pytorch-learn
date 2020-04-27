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
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

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

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


    return train_loader, train_data

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            nn.Sigmoid(),
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


if __name__ == "__main__":
    train_loader, train_data = _load_data_(DOWNLOAD_MNIST)                    # 加载数据

    net = AutoEncoder()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)           # 选取优化器
    loss_func = nn.MSELoss()                                        # 选中损失函数

    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()  # continuously plot

    # original data (first row) for viewing
    view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    # Training
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            b_x = batch_x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
            b_y = batch_x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

            encoded, decoded = net(b_x)                                 # 将数据传入神经网络

            loss = loss_func(decoded, b_y)                           # 计算损失函数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0 :
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

                # plotting decoded image (second row)
                _, decoded_data = net(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())
                plt.draw()
                plt.pause(0.05)

    plt.ioff()
    plt.show()

    # visualize in 3D plot
    view_data = train_data.train_data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    encoded_data, _ = net(view_data)
    fig = plt.figure(2)
    ax = Axes3D(fig)
    X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    values = train_data.train_labels[:200].numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()



