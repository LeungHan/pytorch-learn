import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net,self).__init__()
        # 定义隐藏层节点
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 定义输出层节点
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 输入x
        h = torch.relu(self.hidden(x))
        out = self.predict(h)

        return out



if __name__=="__main__":

    # torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    x, y = Variable(x), Variable(y)


    net = Net(1,10,1)
    # 定义梯度下降算法
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
    loss_func = torch.nn.MSELoss()

    plt.ion()
    plt.show()

    for i in range(100):
        predict = net(x)

        loss = loss_func(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0 :
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw = 5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)


    plt.ioff()
    plt.show()

