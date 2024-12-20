import os

import lovasz_losses as L
import matplotlib.pyplot as plt
import torch
import tqdm
from datapreprocess import *
from loss import *
from net import *
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# BCELoss


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

weight_path = (
    "/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo+den/weights3"
)
data_path = "/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo+den"
save_path = (
    "/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo+den/saveimg3"
)
epoch = 1
epoch_num = 600
continueflag = False

# os.makedirs(save_path, exist_ok=True)

if __name__ == "__main__":
    data_loader = DataLoader(
        dataset=SIMDataset(os.path.join(data_path, "train_data")),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=SIMDataset(os.path.join(data_path, "test_data")),
        batch_size=32,
        shuffle=True,
    )
    net = UNet(1, 3).to(device)
    epoch = 1
    # 判断weight_path是否存在，没有则创建
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "1loss"))
    # 判断weight_path里是否有不同的unet_xx.pth，若有则加载xx最大的pth
    weight_path_list = os.listdir(weight_path)
    if len(weight_path_list) > 0:
        weight_path_list.sort(key=lambda x: int(x[:-4].split("_")[-1]))
        weight_path1 = os.path.join(weight_path, weight_path_list[-1])
        net.load_state_dict(torch.load(weight_path1))
        epoch = int(weight_path_list[-1][:-4].split("_")[-1]) + 1
        continueflag = True
        c = len(os.listdir(os.path.join(save_path, "1loss"))) + 1
        print("load weight: ", weight_path1)
        print("epoch: ", epoch)
    else:
        print("no weight")

    opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-6)
    # loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCELoss()
    # loss_func = FocalLoss()
    # loss_func = DiceLoss()

    train_losses = []
    val_losses = []
    accuracy = []
    diceloss = DiceLoss()
    # print("data_loader length: ", len(data_loader.dataset))

    while epoch < epoch_num:
        for i, (image, mask) in tqdm.tqdm(enumerate(data_loader)):
            image = image.to(device)
            #         mask = mask.long()
            #         mask = mask.squeeze(1)
            mask = mask.float()
            mask = mask.to(device)
            net.train()
            out = net(image)
            # print("out shape: ", out.shape)
            # print("out max and min: ", out.max(), out.min())

            # print("mask shape: ", mask.shape)
            # print("mask max and min: ", mask.max(), mask.min())

            #          print("out shape: ", out.shape)
            # train_loss = loss_func(out,mask)
            train_loss = L.lovasz_softmax(out, mask)

            #            print("train_loss: ", train_loss.item())
            #           print("????")
            opt.zero_grad()
            #           print("zero_grad")
            train_loss.backward()
            #           print("backward")
            opt.step()
            #          print("step")

            if i == 1:
                _image = image[0][0]
                _mask = mask[0][0]
                _out = out[0][2]

                img = torch.cat((_image / 255, _mask, _out), dim=0)
                save_image(img, os.path.join(save_path, f"epoch{epoch}.png"))

        if epoch % 1 == 0:
            val_loss = 0
            iou = 0
            v = 0
            net.eval()
            with torch.no_grad():
                for i, (image, mask) in enumerate(val_loader):
                    image = image.to(device)
                    mask = mask.float()
                    #             mask = mask.squeeze(1)
                    #           mask = mask.long()
                    mask = mask.to(device)
                    out = net(image)

                    val_loss += L.lovasz_softmax(out, mask).item()
                    iou += 0
                    # iou += 1-diceloss(out,mask).item()
                    v += 1

                    if epoch % 5 == 0 and i == 0:
                        _image = image[0][0]
                        _mask = mask[0][0]
                        _out = out[0][2]

                        img = torch.cat((_image / 255, _mask, _out), dim=0)
                        save_image(
                            img,
                            os.path.join(save_path, f"val_epoch{epoch}.png"),
                        )

            val_loss /= v
            iou /= v
            print(
                f"epoch:{epoch},train_loss:{train_loss.item()},val_loss:{val_loss},iou:{iou}"
            )
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            accuracy.append(iou)

            x = range(1, len(train_losses) + 1)
            y1 = train_losses
            y2 = val_losses
            x3 = range(1, len(accuracy) + 1)
            y3 = accuracy
            plt.cla()
            plt.plot(x, y1, color="red", label="train_loss")
            plt.plot(x, y2, color="blue", label="val_loss")
            plt.plot(x3, y3, color="green", label="iou")

            plt.legend()
            if continueflag:
                plt.savefig(
                    os.path.join(save_path, "1loss", f"1loss{c+1}.png")
                )
            else:
                plt.savefig(os.path.join(save_path, "1loss", "1loss1.png"))

        if epoch % 5 == 0:
            torch.save(
                net.state_dict(),
                os.path.join(weight_path, f"unet_{epoch}.pth"),
            )

        epoch += 1

    torch.save(
        net.state_dict(), os.path.join(weight_path, f"unet_{epoch}.pth")
    )
    x = range(1, len(train_losses) + 1)
    y1 = train_losses
    y2 = val_losses
    plt.plot(x, y1, color="red", label="train_loss")
    plt.plot(x, y2, color="blue", label="val_loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "1loss", "1loss.png"))
