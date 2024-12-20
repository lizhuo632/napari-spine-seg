import os

import numpy as np
import tifffile
import torch
from datapreprocess import *
from net import *
from PIL import Image
from torchvision import transforms
from tqdm import trange

# device = torch.device("cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.cuda.empty_cache()
# print('torch.cuda.memory_allocated():',torch.cuda.memory_allocated())
weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/mem-filo-train/weights2/unet_100.pth"  # 195,235

data_paths = [
    # '/share/data/CryoET_Data/lizhuo/SIM-data/AVGtif/3/561.tif',
    #             '/share/data/CryoET_Data/lizhuo/SIM-data/AVGtif/6/6-561-gauss.tif',
    "/share/data/CryoET_Data/lizhuo/SIM-data/mem-filo-train/161415.tif"
]

# data_path = '/share/data/CryoET_Data/lizhuo/SIM-data/traindata/predict'
save_path = "/share/data/CryoET_Data/lizhuo/SIM-data/mem-filo-train"

if not os.path.exists(save_path):
    os.makedirs(save_path)

bin = 2
# 3072x3072
window_size = 256
stride3072 = 192

stride2048 = 128
batch_size = 64
threshold = 0.5


def sliding_window(image, window_size, stride):
    patched = []
    _, h, w = image.shape
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            patched.append(image[:, i : i + window_size, j : j + window_size])
    return np.array(patched)


net = UNet().to(device)
# net = ResUNet().to(device)
net.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
net.eval()
print("load: ", weight_path)

# for file in os.listdir(data_path):
#    if file.endswith('.tif'):
for file in data_paths:
    if file.endswith(".tif"):
        _input = file
        # _input = os.path.join(data_path,file)
        print("input path", _input)
        inputs = tiff.imread(_input)
        print("input shape", inputs.shape)
        if inputs.ndim == 2:
            inputs = np.expand_dims(inputs, axis=0)
        results = np.zeros_like(inputs)
        if inputs.shape[1] == 2048:
            stride = stride2048
        else:
            stride = stride3072
        for t in range(inputs.shape[0]):
            input = Image.fromarray(inputs[t])
            input = transforms.Resize(
                (int(input.size[0] / bin), int(input.size[1] / bin))
            )(input)
            print("input reshape", input.size)
            # reflect padding （window_size-stride）/2
            padded_input = transforms.Pad(
                int((window_size - stride) / 2), fill=0, padding_mode="reflect"
            )(input)
            print("padded_input shape", padded_input.size)
            padded_input = np.reshape(
                padded_input, (1, padded_input.size[0], padded_input.size[1])
            )
            padded_input = np.array(padded_input).astype("float32")
            # image = torch.from_numpy(padded_input)
            print("image shape", padded_input.shape)

            patches = sliding_window(padded_input, window_size, stride)
            print("patches shape", patches.shape)

            patches = torch.from_numpy(patches)
            patches = patches.to(device)
            print("patches shape", patches.shape)
            print(
                "torch.cuda.memory_allocated():", torch.cuda.memory_allocated()
            )

            #           for i in trange(0,int(patches.size()[0]/batch_size)):
            for i in trange(0, 1):
                out = net(
                    patches[i * batch_size : i * batch_size + batch_size]
                )
                # 阈值处理
                # out[out>threshold] = 1
                # out[out<=threshold] = 0

                show = out[0][0].cpu()
                showpatch = patches[i * batch_size][0].cpu()

                # for j in range(1,out.shape[0]):
                #     show = torch.cat((show,out[j][0].cpu()),dim=1)
                #     showpatch = torch.cat((showpatch,patches[i*batch_size+j][0].cpu()),dim=1)
                # #show = show.detach().numpy()
                # tiff.imsave(os.path.join(save_path,file[:-4]+str(weight_path.split('/')[-2])+str(weight_path.split('_')[-1][:-4])+'_result'+str(i)+'.tif'),show.detach().numpy())
                # tiff.imsave(os.path.join(save_path,file[:-4]+str(weight_path.split('/')[-2])+str(weight_path.split('_')[-1][:-4])+'_patch'+str(i)+'.tif'),showpatch.detach().numpy())

                if i == 0:
                    outs = out
                else:
                    outs = torch.cat((outs, out), dim=0)
                # print('torch.cuda.memory_allocated():',torch.cuda.memory_allocated())
                # torch.cuda.empty_cache()
            print("outs shape", outs.shape)
            # out = net(patches)

            # 插值resize out的后两维度
            result = np.zeros_like(input)
            for i in range(outs.shape[0]):
                # 取out的中间部分(stride x stride)作为预测结果,填充到result中
                row = int(i / (int(input.size[0] / stride)))
                col = int(i % (int(input.size[0] / stride)))
                # print("i,row,col",i,row,col)
                result[
                    row * stride : row * stride + stride,
                    col * stride : col * stride + stride,
                ] = (
                    outs[i][0][
                        int((window_size - stride) / 2) : int(
                            (window_size - stride) / 2
                        )
                        + stride,
                        int((window_size - stride) / 2) : int(
                            (window_size - stride) / 2
                        )
                        + stride,
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )

            print("result shape", result.shape)
            result = torch.from_numpy(result)
            result = result.float()
            print("tensor result shape", result.shape)
            result = torch.nn.functional.interpolate(
                result.unsqueeze(0).unsqueeze(0),
                size=(input.size[0] * bin, input.size[1] * bin),
                mode="bilinear",
                align_corners=False,
            )
            print("tensor result reshape", result.shape)

            # result = np.resize(result,(result.shape[0]*bin,result.shape[1]*bin),)

            results[t] = result.detach().numpy()[0][0]

        name = os.path.join(
            save_path, file.split("/")[-1][:-4] + "_mask100.tif"
        )
        results[results > threshold] = 1
        results[results <= threshold] = 0
        tifffile.imsave(name, results)
