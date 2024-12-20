import os

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# import albumentations as A


class SIMDataset(Dataset):
    def __init__(self, path, transform=None):
        self.imgpath = os.path.join(path, "imgs")
        self.maskpath = os.path.join(path, "masks")

        self.imgs = os.listdir(self.imgpath)
        self.imgs.sort()
        self.masks = os.listdir(self.maskpath)
        self.masks.sort()

        self.transform = transform

    def __getitem__(self, idx):
        img_item_name = self.imgs[idx]
        #       print("idx, img_item_name",[idx,img_item_name])
        img_item_path = os.path.join(self.imgpath, img_item_name)
        mask_item_path = os.path.join(
            self.maskpath, img_item_name[:-4] + "_mask.tif"
        )

        # img = Image.open(img_item_path)#.convert('L') #会把65535的uint16转0-255？
        # print("max img",np.max(img))
        # img.show()

        # 直方图均衡化
        # img = ImageOps.equalize(img)

        mask = tiff.imread(mask_item_path)
        mask = mask.astype("uint8")
        mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))
        # 二值化
        mask[mask > 0] = 1

        # imgnp = np.array(img).astype('float32')
        # imgnp = np.array(img).astype('float32')/65535
        # print('max imgnp:',np.max(imgnp))
        imgnp = tiff.imread(img_item_path)
        imgnp = imgnp.astype("float32") / 255
        imgnp = np.reshape(imgnp, (1, imgnp.shape[0], imgnp.shape[1]))
        return torch.from_numpy(imgnp), torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)


transform = transforms.Compose([transforms.ToTensor()])

if __name__ == "__main__":
    data = SIMDataset(
        "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/train_data",
        transform=transform,
    )
    img1, mask1 = data[7]
    print(f"img1:\n{img1},\nmask1:\n{mask1}")
    print(data[49][1].shape)
