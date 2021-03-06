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
            nn.Conv2d(64, 32, (3,3), stride=(2,2), padding=(1,1)),  # b, 32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 16, (3,3), stride=(2,2), padding=(1,1)),  # b, 16, 16, 16
            nn.ReLU(True),
            nn.Conv2d(16, 4, (3,3), stride=(2,2), padding=(1,1)),  # b, 4, 8, 8
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, (3,3), stride=(2,2), padding=(0,0)),  # b, 16, 17, 17
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, (3,3), stride=(2,2), padding=(1,1)),  # b, 32, 33, 33
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, (3,3), stride=(2,2), padding=(1,1)),  # b, 64, 65, 65
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, (4,4), stride=(2,2), padding=(2,2)),  # b, 3, 128, 128
            nn.Tanh()
        )

        # with upsampling
        #self.encoder = nn.Sequential(
        #    nn.Conv2d(3, 64, (5,5), stride=(2,2), padding=(1,1)),  # b, 64, 64, 64
        #    nn.ReLU(True),
        #    nn.Conv2d(64, 32, (3,3), stride=(2,2), padding=(1,1)),  # b, 32, 32, 32
        #    nn.ReLU(True),
        #    nn.Conv2d(32, 16, (3,3), stride=(2,2), padding=(1,1)),  # b, 16, 16, 16
        #    nn.ReLU(True),
        #    nn.Conv2d(16, 4, (3,3), stride=(2,2), padding=(1,1)),  # b, 4, 8, 8
        #    nn.ReLU(True)
        #)
#
        #self.decoder = nn.Sequential(
        #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #    nn.Conv2d(4, 16, (3,3), stride=(1,1), padding=(1,1)),  # b, 16, 16, 16
        #    nn.ReLU(True),
#
        #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #    nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=(1,1)),  # b, 32, 32, 32
        #    nn.ReLU(True),
#
        #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #    nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=(1,1)),  # b, 64, 64, 64
        #    nn.ReLU(True),
#
        #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #    nn.Conv2d(64, 3, (5,5), stride=(1,1), padding=(2,2)),  # b, 3, 128, 128
        #    nn.Tanh()
        #)

    def forward(self, x):
        x = self.encoder(x)       
        # x = self.decoder(x)
        return x

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 128, 128)
    return x

def train_autoencoder(data_file):
    num_epochs = 100
    batch_size = 64
    learning_rate = 1e-4

    dataset = BeproDatasetAutoencoder(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

            if batch_num % 600 == 0:
                pic = to_img(output.cpu().data)
                save_image(pic, './dc_img/image_{}_{}.png'.format(epoch, batch_num))
                print('epoch [{}/{}] batch  [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, batch_num, n_batches, loss.item()))

        ckpt_path = './dc_img/conv_autoencoder_ckpt_{}.pth'.format(epoch+1)
        torch.save(model.state_dict(), ckpt_path)

def infere_autoencoder(data_file, pth_path):

    model = autoencoder().cuda()
    checkpoint = torch.load(pth_path)
    model.load_state_dict(checkpoint)
    dataset = BeproDatasetAutoencoder(data_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_num, data in enumerate(dataloader):
        img = data
        img = Variable(img).cuda()
        output = model.forward(img)
        print ('output size: ' + str(output.size()))
        # pic = to_img(output.cpu().data)
        # save_image(pic, './test_inference.png')

if __name__=="__main__": 
    if sys.argv[2] == 'val':
        infere_autoencoder(sys.argv[1], sys.argv[3])
    elif sys.argv[2] == 'train':
        train_autoencoder(sys.argv[1])
    else:
        print('specify training or inference.')

