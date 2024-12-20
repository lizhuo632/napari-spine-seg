import os

import numpy as np
import tifffile
import torch
from datapreprocess import *
from net import *
from PIL import Image
from skimage import measure
from torchvision import transforms
from tqdm import trange

# device = torch.device("cpu")

# print('torch.cuda.memory_allocated():',torch.cuda.memory_allocated())
# weight for spine+filo。最佳：'/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo/weights0/weig_150.pth' / 200.pth
# weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo/weights0/weig_155.pth'


# weight for dendrite
# weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/traindata/weights3/unet_150.pth'
data_path = "/share/data/CryoET_Data/lizhuo/SIM-data/mem/predict"


################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#################
# 手动归一化输入，imageJ-8bit
# batchsize的调整


def sliding_window(image, window_size, stride):
    patched = []
    _, h, w = image.shape
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            patched.append(image[:, i : i + window_size, j : j + window_size])
    return np.array(patched)


def filter_small(img, threshold):
    # 统计只有0,1二值的results中连续的1值区域，过滤小杂点
    result = np.array(img)
    size_threshold = threshold

    if len(result.shape) == 2:
        # 用measure.label函数将二值图像中的连通区域进行标记
        labeled_image, num_features = measure.label(
            result, connectivity=2, return_num=True
        )

        # 使用measure.regionprops函数获取每个连通区域的属性
        region_props = measure.regionprops(labeled_image)

        i = 1
        # 遍历每个连通区域，根据面积大小进行过滤,并将符合条件的区域的像素值设为不同数字
        for props in region_props:
            if props.area < size_threshold:
                # 将面积小于阈值的区域对应的像素值设为0
                labeled_image[labeled_image == props.label] = 0
            else:
                labeled_image[labeled_image == props.label] = i
                i += 1

        result = labeled_image
    else:
        for i in trange(result.shape[0]):
            i = 1
            labeled_image, num_features = measure.label(
                result[i], connectivity=2, return_num=True
            )
            region_props = measure.regionprops(labeled_image)
            for props in region_props:
                if props.area < size_threshold:
                    labeled_image[labeled_image == props.label] = 0
                else:
                    labeled_image[labeled_image == props.label] = i
                    i += 1
            result[i] = labeled_image

    return result


def predict(data_path, weight_path):
    save_path = data_path + "/result1"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    torch.cuda.empty_cache()
    bin = 2
    # 3072x3072
    window_size = 256
    stride = 192

    stride2048 = 128
    batch_size = 64
    threshold = 0.5

    out_channel = 2
    filter_threshold = 200
    if (
        weight_path.split("/")[-2][-1] == "2"
        or weight_path.split("/")[-2][-1] == "3"
    ):
        out_channel = 1
    net = UNet(1, out_channel).to(device)

    # net = ResUNet().to(device)
    net.load_state_dict(
        torch.load(weight_path, map_location=torch.device("cpu"))
    )
    net.eval()
    print("load: ", weight_path)

    for file in os.listdir(data_path):
        if file.endswith(".tif"):
            _input = os.path.join(data_path, file)
            # print('input path',_input)
            input = Image.open(_input)
            # print('input type',type(input))
            # 归一化0-255
            # image = np.array(image)
            # image = image.astype(np.float32)
            # input = (input - np.min(input)) / (np.max(input) - np.min(input))
            # input = input * 255
            # print('input type',type(input))

            print("input shape", input.size)
            if input.size[0] == 2048:
                stride = stride2048
            input = transforms.Resize(
                (int(input.size[0] / bin), int(input.size[1] / bin))
            )(input)
            # print('input reshape',input.size)
            # reflect padding （window_size-stride）/2
            padded_input = transforms.Pad(
                int((window_size - stride) / 2), fill=0, padding_mode="reflect"
            )(input)
            # print('padded_input shape',padded_input.size)
            padded_input = np.reshape(
                padded_input, (1, padded_input.size[0], padded_input.size[1])
            )
            padded_input = np.array(padded_input).astype("float32")
            # image = torch.from_numpy(padded_input)
            # print('image shape',padded_input.shape)

            patches = sliding_window(padded_input, window_size, stride)
            # print('patches shape',patches.shape)

            patches = torch.from_numpy(patches)
            patches = patches.to(device)
            # print('patches shape',patches.shape)
            # print('torch.cuda.memory_allocated():',torch.cuda.memory_allocated())

            for i in trange(0, int(patches.size()[0] / batch_size)):
                out = net(
                    patches[i * batch_size : i * batch_size + batch_size]
                )
                # 阈值处理
                out[out > threshold] = 1
                out[out <= threshold] = 0

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
            result = np.zeros_like(input)
            # 插值resize out的后两维度

            for i in range(outs.shape[0]):
                # 取out的中间部分(stride x stride)作为预测结果,填充到result中
                row = int(i / (int(input.size[0] / stride)))
                col = int(i % (int(input.size[0] / stride)))
                # print("i,row,col",i,row,col)
                result[
                    row * stride : row * stride + stride,
                    col * stride : col * stride + stride,
                ] = (
                    outs[i][out_channel - 1][
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
                # if out_channel == 2:
                #     result[row*stride:row*stride+stride,col*stride:col*stride+stride] = outs[i][1][int((window_size-stride)/2):int((window_size-stride)/2)+stride,int((window_size-stride)/2):int((window_size-stride)/2)+stride].cpu().detach().numpy()
                # elif out_channel == 3:
                #     result[row*stride:row*stride+stride,col*stride:col*stride+stride] = outs[i][2][int((window_size-stride)/2):int((window_size-stride)/2)+stride,int((window_size-stride)/2):int((window_size-stride)/2)+stride].cpu().detach().numpy()
                # else:
                #     result[row*stride:row*stride+stride,col*stride:col*stride+stride] = outs[i][0][int((window_size-stride)/2):int((window_size-stride)/2)+stride,int((window_size-stride)/2):int((window_size-stride)/2)+stride].cpu().detach().numpy()

            result = filter_small(result, filter_threshold)
            print("result shape", result.shape)
            result = torch.from_numpy(result)
            result = result.float()
            print("tensor result shape", result.shape)
            result = torch.nn.functional.interpolate(
                result.unsqueeze(0).unsqueeze(0),
                size=(input.size[0] * bin, input.size[1] * bin),
                mode="nearest",
            )  # ,align_corners=False)
            print("tensor result reshape", result.shape)

            # result = np.resize(result,(result.shape[0]*bin,result.shape[1]*bin),)
            result255 = result * 255
            name = (
                file[:-4]
                + "_"
                + str(weight_path.split("/")[-2])
                + "_"
                + str(weight_path.split("_")[-1][:-4] + "_result.tif")
            )
            print(name)
            print("result shape", result.shape)
            # tifffile.imsave(os.path.join(save_path,name),result.detach().numpy()[0][0],dtype='uint8')
            # tifffile.imsave(os.path.join(save_path,name[:-4]+'_255.tif'),result255.detach().numpy()[0][0],dtype='uint8')
            tifffile.imwrite(
                os.path.join(save_path, name), result.detach().numpy()[0][0]
            )
            # tifffile.imwrite(os.path.join(save_path,name[:-4]+'_255.tif'),result255.detach().numpy()[0][0])


# for i in range(0,1):

#     weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo/weights'+str(i)
#     for file in os.listdir(weight_path):
#         if file.endswith('.pth'):

#             predict(data_path,os.path.join(weight_path,file))
predict(data_path=data_path, weight_path=weight_path)
