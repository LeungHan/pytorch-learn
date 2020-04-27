import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

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
    # fake dataset
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
    y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # plot dataset
    plt.scatter(x.numpy(), y.numpy())
    #plt.show()

    net_SGD = Net(1, 20, 1)
    net_Momentum = Net(1, 20, 1)
    net_RMSprop = Net(1, 20, 1)
    net_Adam = Net(1, 20, 1)

    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # 定义梯度下降算法
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9,0.99))

    optims = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    loss_his = [[], [], [], []]

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            for net, opt, l_his in zip(nets, optims, loss_his):
                output = net(b_x)
                loss = loss_func(output, b_y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data)

    labels = ["SGD", "Momentum", "RMSprop", "Adam"]
    for i, l_his in enumerate(loss_his):
        plt.plot(l_his, label=labels[i])

    plt.legend(loc="best")
    plt.xlabel("step")
    plt.ylabel("loss")

    plt.ylim((0,0.2))
    plt.show()
