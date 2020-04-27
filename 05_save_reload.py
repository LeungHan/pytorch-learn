import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


# torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)


def _bulid_net():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr = 0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # save
    torch.save(net1, "04_net1.pkl")                          # 保存全部网络模型
    torch.save(net1.state_dict(), "04_net1_params.pkl")      # 保存网络模型参数

def _restore_net():
    net2 = torch.load("04_net1.pkl")
    prediction = net2(x)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def _restore_params():
    # 搭建同意的网络结构
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    #加载网络模型参数
    net3.load_state_dict(torch.load("04_net1_params.pkl"))
    prediction = net3(x)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


if __name__ == "__main__":
    _bulid_net()
    _restore_net()
    _restore_params()
    plt.show()





