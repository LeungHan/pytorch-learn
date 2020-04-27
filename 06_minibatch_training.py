import torch
import torch.utils.data as DATA

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5

# fake data
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_data = DATA.TensorDataset(x, y)

loader = DATA.DataLoader(
    dataset = torch_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2
)

def show_patch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print("Epoch:", epoch, "|Step:", step, "|batch_x:", batch_x.numpy(), "|batch_y:", batch_y.numpy())


if __name__ == "__main__":
    show_patch()

