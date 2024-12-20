import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from datapreprocess import *
from loss import *
from net import *
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/traindata/weights"
data_path = "/share/data/CryoET_Data/lizhuo/SIM-data/traindata"
save_path = "/share/data/CryoET_Data/lizhuo/SIM-data/traindata/saveimg"
net = UNet().to(device)
net1 = UNet(1, 2).to(device)
val_losses = []
diceloss = DiceLoss()
for i in range(1, 4):
    val_losses = []
    for j in range(1, 61):
        weight_pathij = os.path.join(weight_path + str(i), f"unet_{j*5}.pth")
        print(weight_pathij)
        if i == 1:
            net1.load_state_dict(torch.load(weight_pathij))
        else:
            net.load_state_dict(torch.load(weight_pathij))

        val_loader = DataLoader(
            dataset=SIMDataset(os.path.join(data_path, "test_data")),
            batch_size=32,
            shuffle=True,
        )
        net.eval()
        net1.eval()
        val_loss = 0
        kk = 0
        with torch.no_grad():
            for k, (image, mask) in tqdm.tqdm(enumerate(val_loader)):
                image = image.to(device)
                mask = mask.float()
                mask = mask.to(device)
                if i == 1:
                    out = net1(image)
                    out2 = out[:, 1, :, :]
                    out2reshape = torch.reshape(
                        out2, (out2.shape[0], 1, out2.shape[1], out2.shape[2])
                    )
                    # print('outshape:',out.shape)
                    # print('outshape:',out2reshape.shape)
                    # print('maskshape:',mask.shape)
                    val_loss += diceloss(out2reshape, mask).item()
                    kk += 1
                else:
                    out = net(image)

                    val_loss += diceloss(out, mask).item()
                    kk += 1
        val_loss /= kk
        val_losses.append(val_loss)
        x = np.arange(1, len(val_losses) + 1)
        y = np.array(val_losses)
        plt.cla()
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("val_loss")
        plt.legend()
        plt.savefig(os.path.join(save_path + str(i), "1loss", "dice_loss.png"))


# net = UNet().to(device)
# if os.path.exists(weight_path):
#     net.load_state_dict(torch.load(weight_path))
#     print('load weight success')
# else:
#     print('no weight')

# transform = transforms.Compose([
#     transforms.Resize((512,512)),
# ])
# img = Image.open(_input).convert('L')
# img = transform(img)
# imgnp = np.array(img).astype('float32')
# imgnp = np.reshape(imgnp,(1,1,imgnp.shape[0],imgnp.shape[1]))
# imgnp = torch.from_numpy(imgnp)

# out = net(imgnp.to(device))

# result = torch.cat(((imgnp[0][0]/255).cpu(),out[0][0].cpu()),dim=1)
# save_image(result, os.path.join(save_path, '7.png'))
