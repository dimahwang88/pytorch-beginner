__author__ = 'dimahwang'

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import os
import sys
from PIL import Image

data_file = sys.argv[1]

class BeproDatasetAutoencoder(Dataset):
    def __init__(self, img_txt_path):
        self.data_info = []

        f = open(img_txt_path, 'r')
        for path in f:
            path = path[:-1]
            if not os.path.isfile(path):
                print('depricated: ' + path)
                continue
            self.data_info.append(path)

        self.to_tensor = transforms.ToTensor()
        self.to_normal = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.to_resize = transforms.Resize((28,28))
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        single_image_name = self.data_info[index]
        img_as_img = Image.open(single_image_name)
        img_as_resize = self.to_resize(img_as_img)
        img_as_tensor = self.to_tensor(img_as_resize)
        img_as_norm = self.to_normal(img_as_tensor)
        return img_as_norm

    def __len__(self):
        return self.data_len

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 32
learning_rate = 1e-3

dataset = BeproDatasetAutoencoder(data_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))

#    if epoch % 10 == 0:
#        pic = to_img(output.cpu().data)
#        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')