import os

import matplotlib.pyplot as plt
import torch
import tqdm
from datapreprocess import *
from loss import *
from net import *
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# CELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train10/weights2"
data_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train10"
save_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train10/saveimg2"
epoch = 1
epoch_num = 500
continueflag = False
# loss = 'ce' #ce focal dice
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
    net = UNet(1, 2).to(device)
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
        c = len(os.listdir(os.path.join(save_path, "1loss"))) + 1
        continueflag = True
        print("load weight: ", weight_path1)
        print("epoch: ", epoch)
    else:
        print("no weight")

    opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-6)
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCELoss()
    # loss_func = FocalLoss()
    # loss_func = DiceLoss()

    train_losses = []
    val_losses = []
    ratess = []
    ratess2 = []
    diceloss = DiceLoss()
    # print("data_loader length: ", len(data_loader.dataset))

    threshold = 0.5

    while epoch < epoch_num:
        for i, (image, mask) in tqdm.tqdm(enumerate(data_loader)):
            image = image.to(device)
            mask = mask.long()
            mask = mask.squeeze(1)
            #          mask = mask.float()
            mask = mask.to(device)
            net.train()
            out = net(image)
            # print("out shape: ", out.shape)
            # print("out max and min: ", out.max(), out.min())

            # print("mask shape: ", mask.shape)
            # print("mask max and min: ", mask.max(), mask.min())

            #          print("out shape: ", out.shape)
            train_loss = loss_func(out, mask)

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
                _mask = mask[0]
                _out = out[0][1]
                _out_threshold = torch.where(
                    _out > threshold,
                    torch.ones_like(_out),
                    torch.zeros_like(_out),
                )
                print("IOU:", get_IOU(_out, _mask, threshold))
                rates2 = [
                    get_accuracy(_out, _mask, threshold),
                    get_precision(_out, _mask, threshold),
                    get_recall(_out, _mask, threshold),
                    get_F1(_out, _mask, threshold),
                    get_IOU(_out, _mask, threshold),
                ]
                ratess2.append(rates2)

                img = torch.cat((_image, _mask, _out, _out_threshold), dim=0)
                save_image(img, os.path.join(save_path, f"epoch{epoch}.png"))

        if epoch % 1 == 0:
            val_loss = 0
            v = 0
            iou = 0

            # acc,precision,recall,F1,IOU
            rates = [0, 0, 0, 0, 0]
            net.eval()
            with torch.no_grad():
                for i, (image, mask) in enumerate(val_loader):
                    image = image.to(device)
                    # mask = mask.float()
                    mask = mask.squeeze(1)
                    mask = mask.long()
                    mask = mask.to(device)
                    out = net(image)

                    # if out.size()[1] == 2:
                    #     _out = out[:,1,:,:]
                    #     _out = _out.squeeze(1)
                    _out = out[0][1]
                    _mask = mask[0]

                    val_loss += loss_func(out, mask).item()
                    iou += 1 - diceloss(out, mask).item()
                    rates[0] += get_accuracy(_out, _mask, threshold)
                    rates[1] += get_precision(_out, _mask, threshold)
                    rates[2] += get_recall(_out, _mask, threshold)
                    rates[3] += get_F1(_out, _mask, threshold)
                    rates[4] += get_IOU(_out, _mask, threshold)

                    v += 1

                    if epoch % 1 == 0 and i == 0:
                        _image = image[0][0]
                        _mask = mask[0]
                        _out = out[0][1]
                        _out_threshold = torch.where(
                            _out > threshold,
                            torch.ones_like(_out),
                            torch.zeros_like(_out),
                        )

                        img = torch.cat(
                            (_image, _mask, _out, _out_threshold), dim=0
                        )
                        save_image(
                            img,
                            os.path.join(save_path, f"val_epoch{epoch}.png"),
                        )

            val_loss /= v
            iou /= v
            rates = [rate / v for rate in rates]
            print(
                f"epoch:{epoch},train_loss:{train_loss.item()},val_loss:{val_loss},iou:{iou}"
            )
            print(
                f"val:acc:{rates[0]},precision:{rates[1]},recall:{rates[2]},F1:{rates[3]},IOU:{rates[4]}"
            )
            print(
                f"train:acc2:{rates2[0]},precision2:{rates2[1]},recall2:{rates2[2]},F12:{rates2[3]},IOU2:{rates2[4]}"
            )
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            ratess.append(rates)

            x = range(1, len(train_losses) + 1)
            y1 = train_losses
            y2 = val_losses
            x3 = range(1, len(ratess) + 1)
            y3 = [rate[0] for rate in ratess]
            y4 = [rate[1] for rate in ratess]
            y5 = [rate[2] for rate in ratess]
            y6 = [rate[3] for rate in ratess]
            y7 = [rate[4] for rate in ratess]
            x33 = range(1, len(ratess2) + 1)
            y33 = [rate[0] for rate in ratess2]
            y44 = [rate[1] for rate in ratess2]
            y55 = [rate[2] for rate in ratess2]
            y66 = [rate[3] for rate in ratess2]
            y77 = [rate[4] for rate in ratess2]
            # fig = plt.figure()
            plt.cla()
            plt.plot(x, y1, color="red", label="Train_loss")
            plt.plot(x, y2, color="blue", label="Val_loss")
            plt.plot(x33, y33, color="green", label="Accuracy")
            plt.plot(x33, y44, color="yellow", label="Precision")
            plt.plot(x33, y55, color="purple", label="Recall")
            plt.plot(x33, y66, color="black", label="F1-Score")
            plt.plot(x33, y77, color="orange", label="IOU")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "1loss", "2loss2.png"))
            plt.cla()
            plt.plot(x, y1, color="red", label="Train_loss")
            plt.plot(x, y2, color="blue", label="Val_loss")
            plt.plot(x3, y3, color="green", label="Accuracy")
            plt.plot(x3, y4, color="yellow", label="Precision")
            plt.plot(x3, y5, color="purple", label="Recall")
            plt.plot(x3, y6, color="black", label="F1-Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "1loss", "noIOUloss1.png"))
            plt.plot(x3, y7, color="orange", label="IOU")
            plt.legend()

            if continueflag:

                plt.savefig(os.path.join(save_path, "1loss", f"1loss{c}.png"))
            else:
                plt.savefig(os.path.join(save_path, "1loss", "1loss1.png"))

        if epoch % 1 == 0:
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
