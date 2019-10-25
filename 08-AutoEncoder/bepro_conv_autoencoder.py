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
        self.to_resize = transforms.Resize((128,128))
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
            nn.Conv2d(3, 64, (5,5), stride=(2,2), padding=(2,2)),  # b, 64, 64, 64
            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=(2,2)),  # b, 16, 32, 32
            nn.Conv2d(64, 32, (3,3), stride=(2,2), padding=(1,1)),  # b, 32, 32, 32
            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=(2,2)),  # b, 8, 8, 8
            nn.Conv2d(32, 16, (3,3), stride=(2,2), padding=(1,1)),  # b, 16, 16, 16
            nn.ReLU(True),
            nn.Conv2d(16, 4, (3,3), stride=(2,2), padding=(1,1)),  # b, 4, 8, 8
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, (3,3), stride=(2,2), padding=(1,1)),  # b, 16, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, (3,3), stride=(2,2), padding=(1,1)),  # b, 32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, (3,3), stride=(2,2), padding=(1,1)),  # b, 64, 64, 64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, (5,5), stride=(2,2), padding=(2,2)),  # b, 3, 128, 128            
            nn.Tanh()
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
            print(x.size())
        
        for layer in self.decoder:
            x = layer(x)
            print(x.size())

        #x = self.encoder(x)       
        #x = self.decoder(x)
        return x

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 128, 128)
    return x

num_epochs = 100
batch_size = 64
learning_rate = 1e-3

dataset = BeproDatasetAutoencoder(data_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

n_batches = int(len(dataloader.dataset) / batch_size)

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

for epoch in range(num_epochs):
    for batch_num, data in enumerate(dataloader):
        img = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/image_{}_{}.png'.format(epoch, batch_num))
            print('epoch [{}/{}] batch  [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, batch_num, n_batches, loss.item()))
        
    #print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))

torch.save(model.state_dict(), './conv_autoencoder.pth')