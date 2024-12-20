# 集群condadl运行

import os
from functools import partial
from operator import itemgetter
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from datapreprocess import *
from loss import *
from net import *
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import class2one_hot, one_hot2dist

D = Union[Image.Image, np.ndarray, Tensor]


def gt_transform(
    resolution: Tuple[float, ...], K: int
) -> Callable[[D], Tensor]:
    return transforms.Compose(
        [
            lambda img: np.array(img)[...],
            # lambda nd: torch.tensor(nd, dtype=torch.int64)[:, None, ...],
            lambda nd: torch.tensor(nd, dtype=torch.int64)[
                None, ...
            ],  # Add one dimension to simulate batch
            partial(class2one_hot, K=K),
            itemgetter(0),  # Then pop the element to go back to img shape
        ]
    )


def dist_map_transform(
    resolution: Tuple[float, ...], K: int
) -> Callable[[D], Tensor]:
    return transforms.Compose(
        [
            gt_transform(resolution, K),
            lambda t: t.cpu().numpy(),
            partial(one_hot2dist, resolution=resolution),
            lambda nd: torch.tensor(nd, dtype=torch.float32),
        ]
    )


def gt_transform_batch(
    resolution: Tuple[float, ...], K: int
) -> Callable[[np.ndarray], torch.Tensor]:
    """
    Process batch of masks [batchsize, w, h] to one-hot encoded form [batchsize, K, w, h].
    """

    def transform(masks: np.ndarray):
        batch = []
        for mask in masks:  # Loop through each sample in the batch
            one_hot_mask = class2one_hot(
                torch.tensor(mask, dtype=torch.int64)[None, ...], K=K
            )
            batch.append(
                one_hot_mask[0]
            )  # Remove batch dim added temporarily for single image processing
        return torch.stack(batch, dim=0)  # Combine back into batch

    return transform


def dist_map_transform_batch(
    resolution: Tuple[float, ...], K: int
) -> Callable[[np.ndarray], torch.Tensor]:
    """
    Process batch of masks [batchsize, w, h] to distance maps [batchsize, K, w, h].
    """

    def transform(masks: np.ndarray):
        batch = []
        for mask in masks:  # Loop through each sample in the batch
            one_hot_mask = class2one_hot(
                torch.tensor(mask, dtype=torch.int64)[None, ...], K=K
            )
            one_hot_np = (
                one_hot_mask[0].cpu().numpy()
            )  # Convert to numpy array
            # print('one_hot_np',one_hot_np)
            # print('one_hot_np shape:',one_hot_np.shape)

            dist_map = one_hot2dist(one_hot_np, resolution=resolution)
            # print('dist_map',dist_map)
            # print('dist_map shape:',dist_map.shape)
            batch.append(torch.tensor(dist_map, dtype=torch.float32))
            # img = torch.cat((one_hot_mask[0][0],Tensor(dist_map[0])),dim=0)
            # print('dist_map max,min:',Tensor(dist_map[0]).max(),Tensor(dist_map[0]).min())
            # save_image(img,'/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13/saveimg2/2.png')
            # exit()
        return torch.stack(batch, dim=0)  # Combine back into batch

    return transform


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# CELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13/weights2"
data_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13"
save_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13/saveimg2"

loss_type = "focal"  #'focal'
if loss_type == "ce":
    loss_func = (
        nn.CrossEntropyLoss()
    )  # mask shape:  torch.Size([32, 256, 256]) out shape:  torch.Size([32, 2, 256, 256])
    logits = True  # 未归一化
elif loss_type == "focal":
    loss_func = FocalLoss()  # target=[batch,channel,height,width]
    logits = False
elif loss_type == "dice":
    loss_func = DiceLoss()
else:
    raise ValueError("Invalid loss type")

epoch = 1
epoch_num = 200
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
    net = UNet(1, 2, logits=logits).to(device)
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
    # opt = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-5)

    train_losses = []
    val_losses = []
    # ratess = []
    # ratess2 = []
    train_ratess = []
    val_ratess = []
    diceloss = DiceLoss()
    # print("data_loader length: ", len(data_loader.dataset))

    threshold = 0.5

    while epoch < epoch_num:
        for i, (image, mask) in tqdm.tqdm(enumerate(data_loader)):
            image = image.to(device)
            # print('i:',i,'imageshape:',image.shape)
            mask = mask.long()
            mask = mask.squeeze(1)
            #          mask = mask.float()
            mask = mask.to(device)
            net.train()
            out = net(
                image
            )  # mask shape:  torch.Size([32, 256, 256]) out shape:  torch.Size([32, 2, 256, 256])

            # print("out shape: ", out.shape)
            # print("out max and min: ", out.max(), out.min())

            # print("mask shape: ", mask.shape)
            # print("mask max and min: ", mask.max(), mask.min())

            # print("out shape: ", out.shape)

            train_loss = loss_func(out, mask)
            print("train_loss: ", train_loss.item())

            # SurfaceLoss
            mask_dist_map = dist_map_transform_batch((1, 1), 2)(
                Tensor.cpu(mask)
            )
            # print(Tensor.cpu(mask).shape)
            # print(mask_dist_map.shape)
            # 把当前已经sigmoid的out转softmax
            out_one_hot = F.softmax(out, dim=1)

            surface_loss = SurfaceLoss(idc=[1])(
                out_one_hot, mask_dist_map.to(device)
            )
            print("surface_loss: ", surface_loss.item())

            train_loss += surface_loss * 0.01

            #            print("train_loss: ", train_loss.item())
            #           print("????")
            opt.zero_grad()
            #           print("zero_grad")
            train_loss.backward()
            #           print("backward")
            opt.step()
            #          print("step")

            if i == 0:

                avg_train_rates = [0, 0, 0, 0, 0]
                if logits:
                    out_probs = torch.sigmoid(out)
                else:
                    out_probs = out
                for j in range(len(image)):
                    _image = image[j][0]
                    _mask = mask[j]
                    _out_probs = out_probs[j][1]
                    _out_threshold = torch.where(
                        _out_probs > threshold,
                        torch.ones_like(_out_probs),
                        torch.zeros_like(_out_probs),
                    )
                    avg_train_rates[0] += get_accuracy(
                        _out_probs, _mask, threshold
                    )
                    avg_train_rates[1] += get_precision(
                        _out_probs, _mask, threshold
                    )
                    avg_train_rates[2] += get_recall(
                        _out_probs, _mask, threshold
                    )
                    avg_train_rates[3] += get_F1(_out_probs, _mask, threshold)
                    avg_train_rates[4] += get_IOU(_out_probs, _mask, threshold)

                    if j == 0:
                        img = torch.cat(
                            (_image, _mask, _out_probs, _out_threshold), dim=0
                        )
                        save_image(
                            img,
                            os.path.join(save_path, f"epoch{epoch}_{j}.png"),
                        )

                avg_train_rates = [
                    rate / len(image) for rate in avg_train_rates
                ]
                print(
                    f"AVG train rates in {len(image)} img: acc:{avg_train_rates[0]}, precision:{avg_train_rates[1]}, recall:{avg_train_rates[2]}, F1:{avg_train_rates[3]}, IOU:{avg_train_rates[4]}"
                )
                train_ratess.append(avg_train_rates)
                # 改val ratess和plt

                # _image = image[0][0]
                # _mask = mask[0]
                # _out = out[0][1]
                # _out_threshold = torch.where(_out>threshold,torch.ones_like(_out),torch.zeros_like(_out))
                # print("IOU:",get_IOU(_out,_mask,threshold))
                # rates2 = [get_accuracy(_out,_mask,threshold),get_precision(_out,_mask,threshold),get_recall(_out,_mask,threshold),get_F1(_out,_mask,threshold),get_IOU(_out,_mask,threshold)]
                # ratess2.append(rates2)

                # img = torch.cat((_image*255,_mask,_out,_out_threshold),dim=0)
                # save_image(img,os.path.join(save_path,'epoch{}.png'.format(epoch)))

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
                    # _out_probs = out[0][1]
                    # _mask = mask[0]

                    val_loss += loss_func(out, mask).item()
                    v += 1
                    if v == 1:
                        avg_val_rates = [0, 0, 0, 0, 0]
                        if logits:
                            out_probs = torch.sigmoid(out)
                        else:
                            out_probs = out

                        for j in range(len(image)):
                            _image = image[j][0]
                            _mask = mask[j]
                            _out_probs = out_probs[j][1]
                            _out_threshold = torch.where(
                                _out_probs > threshold,
                                torch.ones_like(_out_probs),
                                torch.zeros_like(_out_probs),
                            )
                            avg_val_rates[0] += get_accuracy(
                                _out_probs, _mask, threshold
                            )
                            avg_val_rates[1] += get_precision(
                                _out_probs, _mask, threshold
                            )
                            avg_val_rates[2] += get_recall(
                                _out_probs, _mask, threshold
                            )
                            avg_val_rates[3] += get_F1(
                                _out_probs, _mask, threshold
                            )
                            avg_val_rates[4] += get_IOU(
                                _out_probs, _mask, threshold
                            )

                            # if j == 0:
                            #     img = torch.cat((_image * 255, _mask, _out, _out_threshold), dim=0)
                            #     save_image(img, os.path.join(save_path, 'val_epoch{}_{}.png'.format(epoch, j)))

                        avg_val_rates = [
                            rate / len(image) for rate in avg_val_rates
                        ]
                        print(
                            f"AVG val rates in {len(image)} img: acc:{avg_val_rates[0]}, precision:{avg_val_rates[1]}, recall:{avg_val_rates[2]}, F1:{avg_val_rates[3]}, IOU:{avg_val_rates[4]}"
                        )
                        val_ratess.append(avg_val_rates)
                        _image = image[0][0]
                        _mask = mask[0]
                        _out_probs = out[0][1]
                        _out_threshold = torch.where(
                            _out_probs > threshold,
                            torch.ones_like(_out_probs),
                            torch.zeros_like(_out_probs),
                        )

                        img = torch.cat(
                            (_image, _mask, _out_probs, _out_threshold), dim=0
                        )
                        save_image(
                            img,
                            os.path.join(
                                save_path,
                                f"val_epoch{epoch}_iou{avg_val_rates[4]}.png",
                            ),
                        )

                    # iou += 1-diceloss(out,mask).item()
                    # rates[0] += get_accuracy(_out,_mask,threshold)
                    # rates[1] += get_precision(_out,_mask,threshold)
                    # rates[2] += get_recall(_out,_mask,threshold)
                    # rates[3] += get_F1(_out,_mask,threshold)
                    # rates[4] += get_IOU(_out,_mask,threshold)

                    # v += 1

                    # if epoch % 1 == 0 and i == 0:
                    #     _image = image[0][0]
                    #     _mask = mask[0]
                    #     _out = out[0][1]
                    #     _out_threshold = torch.where(_out>threshold,torch.ones_like(_out),torch.zeros_like(_out))

                    #     img = torch.cat((_image*255,_mask,_out,_out_threshold),dim=0)
                    #     save_image(img,os.path.join(save_path,'val_epoch{}.png'.format(epoch)))

            val_loss /= v
            iou = avg_train_rates[4]
            # rates = [rate/v for rate in rates]
            print(
                f"epoch:{epoch},train_loss:{train_loss.item()},val_loss:{val_loss},iou:{iou}"
            )
            # print('val:acc:{},precision:{},recall:{},F1:{},IOU:{}'.format(rates[0],rates[1],rates[2],rates[3],rates[4]))
            # print('train:acc2:{},precision2:{},recall2:{},F12:{},IOU2:{}'.format(rates2[0],rates2[1],rates2[2],rates2[3],rates2[4]))
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            # ratess.append(rates)

            x_train = range(1, len(train_losses) + 1)
            x_train_ratess = range(1, len(train_ratess) + 1)
            x_val = range(1, len(val_losses) + 1)
            x_val_ratess = range(1, len(val_ratess) + 1)

            y_train_loss = train_losses
            y_val_loss = val_losses

            y_train_acc = [rate[0] for rate in train_ratess]
            y_train_precision = [rate[1] for rate in train_ratess]
            y_train_recall = [rate[2] for rate in train_ratess]
            y_train_F1 = [rate[3] for rate in train_ratess]
            y_train_IOU = [rate[4] for rate in train_ratess]

            y_val_acc = [rate[0] for rate in val_ratess]
            y_val_precision = [rate[1] for rate in val_ratess]
            y_val_recall = [rate[2] for rate in val_ratess]
            y_val_F1 = [rate[3] for rate in val_ratess]
            y_val_IOU = [rate[4] for rate in val_ratess]

            plt.cla()
            plt.plot(x_train, y_train_loss, color="red", label="Train_loss")
            plt.plot(x_val, y_val_loss, color="blue", label="Val_loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "1loss", "loss.png"))

            plt.cla()
            plt.plot(
                x_train_ratess, y_train_acc, color="green", label="Train_acc"
            )
            plt.plot(
                x_train_ratess,
                y_train_precision,
                color="yellow",
                label="Train_precision",
            )
            plt.plot(
                x_train_ratess,
                y_train_recall,
                color="purple",
                label="Train_recall",
            )
            plt.plot(
                x_train_ratess, y_train_F1, color="black", label="Train_F1"
            )
            plt.plot(
                x_train_ratess, y_train_IOU, color="orange", label="Train_IOU"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "1loss", "train_score.png"))

            plt.cla()
            plt.plot(x_val_ratess, y_val_acc, color="green", label="Val_acc")
            plt.plot(
                x_val_ratess,
                y_val_precision,
                color="yellow",
                label="Val_precision",
            )
            plt.plot(
                x_val_ratess, y_val_recall, color="purple", label="Val_recall"
            )
            plt.plot(x_val_ratess, y_val_F1, color="black", label="Val_F1")
            plt.plot(x_val_ratess, y_val_IOU, color="orange", label="Val_IOU")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "1loss", "val_score.png"))

            with open(
                os.path.join(save_path, "1loss", "loss&score.txt"), "a"
            ) as f:
                f.write(
                    f"epoch:{epoch},train_loss:{train_loss.item()},val_loss:{val_loss},iou:{iou}\n"
                )
                f.write(
                    f"train:acc:{avg_train_rates[0]},precision:{avg_train_rates[1]},recall:{avg_train_rates[2]},F1:{avg_train_rates[3]},IOU:{avg_train_rates[4]}\n"
                )
                f.write(
                    f"val:acc:{avg_val_rates[0]},precision:{avg_val_rates[1]},recall:{avg_val_rates[2]},F1:{avg_val_rates[3]},IOU:{avg_val_rates[4]}\n"
                )

            # x = range(1,len(train_losses)+1)
            # y1 = train_losses
            # y2 = val_losses
            # x3 = range(1,len(ratess)+1)
            # y3 = [rate[0] for rate in ratess]
            # y4 = [rate[1] for rate in ratess]
            # y5 = [rate[2] for rate in ratess]
            # y6 = [rate[3] for rate in ratess]
            # y7 = [rate[4] for rate in ratess]
            # x33 = range(1,len(ratess2)+1)
            # y33 = [rate[0] for rate in ratess2]
            # y44 = [rate[1] for rate in ratess2]
            # y55 = [rate[2] for rate in ratess2]
            # y66 = [rate[3] for rate in ratess2]
            # y77 = [rate[4] for rate in ratess2]
            # #fig = plt.figure()
            # plt.cla()
            # plt.plot(x,y1,color='red',label='Train_loss')
            # plt.plot(x,y2,color='blue',label='Val_loss')
            # plt.plot(x33,y33,color='green',label='Accuracy')
            # plt.plot(x33,y44,color='yellow',label='Precision')
            # plt.plot(x33,y55,color='purple',label='Recall')
            # plt.plot(x33,y66,color='black',label='F1-Score')
            # plt.plot(x33,y77,color='orange',label='IOU')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(os.path.join(save_path,'1loss','loss+train_score.png'))
            # plt.cla()
            # plt.plot(x,y1,color='red',label='Train_loss')
            # plt.plot(x,y2,color='blue',label='Val_loss')
            # plt.plot(x3,y3,color='green',label='Accuracy')
            # plt.plot(x3,y4,color='yellow',label='Precision')
            # plt.plot(x3,y5,color='purple',label='Recall')
            # plt.plot(x3,y6,color='black',label='F1-Score')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(os.path.join(save_path,'1loss','loss+val_score+noIOU.png'))
            # plt.plot(x3,y7,color='orange',label='IOU')
            # plt.legend()

            # if continueflag:

            #     plt.savefig(os.path.join(save_path,'1loss','loss{}+val_score.png'.format(c)))
            # else:
            #     plt.savefig(os.path.join(save_path,'1loss','1loss1.png'))

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
