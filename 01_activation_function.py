import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch.nn.functional as F

# fake data
x = torch.linspace(-5,5,200)        #x data (tensor),shape(200,1)
x_np = x.data.numpy()

y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = torch.softmax(x).data.numpy()

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np, y_relu, color="red", label="relu")
plt.ylim((-1,5))
plt.legend(loc="best")

plt.subplot(222)
plt.plot(x_np, y_sigmoid, color="red", label="sigmoid")
plt.ylim((-0.2,1.2))
plt.legend(loc="best")

plt.subplot(223)
plt.plot(x_np, y_tanh, color="red", label="tanh")
plt.ylim((-1.2,1.2))
plt.legend(loc="best")

plt.subplot(224)
plt.plot(x_np, y_softplus, color="red", label="softplus")
plt.ylim((-0.2,6))
plt.legend(loc="best")

plt.show()