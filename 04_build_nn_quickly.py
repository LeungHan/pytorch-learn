import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


if __name__ == "__main__":

    # load data
    n_data = torch.ones(100,2)
    x0 = torch.normal(2*n_data,1)
    y0 = torch.zeros(100)               #标签0

    x1 = torch.normal(-2*n_data,1)
    y1 = torch.ones(100)                #标签1

    x = torch.cat((x0,x1),0).type(torch.FloatTensor)
    y = torch.cat((y0, y1), 0).type(torch.LongTensor)

    x,y = Variable(x),Variable(y)

    net = torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,2)
    )

    optimizer = torch.optim.SGD(net.parameters(), lr = 0.02)
    loss_func = torch.nn.CrossEntropyLoss()

    for i in range(100):
        out = net(x)

        loss = loss_func(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 2 == 0:
            plt.cla()
            # torch.max(a,1)：返回a中每一行中最大值的那个元素
            # troch.max()[1]， 只返回最大值的每个索引
            # dim=0表示按列计算；dim=1表示按行计算
            tmp = torch.softmax(out, dim=1)
            predict = torch.max(tmp, 1)[1]  # 获取每一行最大值的索引
            pred_y = predict.data.numpy()
            target_y = y.data.numpy()

            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()
