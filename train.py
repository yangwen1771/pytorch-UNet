import os

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'
if __name__ == '__main__':
    #num_classes = 2 + 1  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)
    net = UNet().to(device)                                                           #UNet(num_classes).to(device)
    if os.path.exists(weight_path):                                                   #？
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()                                              #loss_fun = nn.CrossEntropyLoss()

    epoch = 1
    while epoch < 200:                                                            #一直循环while True: 再手动停止
        for i, (image, segment_image) in enumerate(data_loader):                  #enumerate(tqdm.tqdm(data_loader))
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)                       #segment_image.long()
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        if epoch % 50 == 0:                                                      #每隔50批次，保存一个权重
            torch.save(net.state_dict(), weight_path)                            #放入net的一个参数，保存位置为weight_path
            print('save successfully!')

        _image=image[0]
        _segement_image=segement_image[0]
        _out_image=out_image[0]
        img=torch.stack([_image,_segement_image,_out_image],dim=0)
        save_image(img,f'{save_path}/{i}.png')                                     #不是只给地址，而是给文件名称
        
        epoch += 1
