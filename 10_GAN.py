import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001               # learning rate for generator
LR_D = 0.0001               # learning rate for generator
N_IDEAS = 5                 # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15         # it could be total point G can draw in the canvas


# np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
# np.linspace(-1,1,ART_COMPONENTS): 生成ART_COMPONENTS个（-1,1）的数据
# 生成64组一样的数据点
PAINT_POINTS  = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])                    # PAINT_POINTS.shape = (64,15)
# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_works():     # painting from the famous artist (real target)
    # numpy.random.uniform(low,high,size):从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    # np.newaxis的作用就是在这一位置增加一个一维
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]                 # 增加到二维
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()

    return paintings


if __name__ == "__main__":
    G = nn.Sequential(
        nn.Linear(N_IDEAS, 128),
        nn.ReLU(),
        nn.Linear(128, ART_COMPONENTS)
    )

    D = nn.Sequential(
        nn.Linear(ART_COMPONENTS, 128),
        nn.ReLU(),
        nn.Linear(128, N_IDEAS),
        nn.Sigmoid()
    )

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)

    plt.ion()  # something about continuous plotting
    plt.show()

    for step in range(10000):
        artist_paintings = artist_works()           # 生成标准数据

        # bulid network
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
        G_output = G(G_ideas)

        D_output1 = D(artist_paintings)
        D_output2 = D(G_output)

        G_loss = torch.mean(torch.log(1. - D_output2))
        D_loss = - torch.mean(torch.log(D_output1) + torch.log(1. - D_output2))

        # Train
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if step % 50 == 0:  # plotting
            plt.cla()
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.plot(PAINT_POINTS[0], G_output.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % D_output1.data.numpy().mean(),
                     fontdict={'size': 13})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=10)
            plt.draw()
            plt.pause(0.01)

    plt.ioff()
    plt.show()





