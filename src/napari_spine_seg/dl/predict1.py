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


type = "mem"
type = "dend"
type = "actin"


data_path = "/share/data/CryoET_Data/lizhuo/wenlan_data/alpha-syn/predict/test3"  #'/share/data/CryoET_Data/lizhuo/SIM-data/240118/2/predict'#'/share/data/CryoET_Data/lizhuo/test/predicttest/class'#'/share/data/CryoET_Data/lizhuo/preprocessed_data/synap488_actin561_psd640/#20231201_1113culture_Mg-ctrl_cLTP-coverslip2/roi2/test2'#'/share/data/CryoET_Data/lizhuo/work_with_wenlan/0423data/singleframetest' #wenlan0307/test6' #/SIM-data/240118/2_norm # data/
# data_path = '/share/data/CryoET_Data/lizhuo/work_with_wenlan/cLTP_data/ylq_cLTP-20min-5min-1h_DATA/5'


if type == "actin":
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train1/weights1/unet_400.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train3/weights1/unet_134.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train4/weights1/unet_100.pth"
    # weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train10/weights2/unet_202.pth'
    # weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train10/weights2/unet_109.pth'
    # weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train5/weights1/unet_100.pth'
    filter_threshold = 50  # 50#25 #bin2后
    out_channel = 2
elif type == "mem":
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo/weights0/weig_200.pth"  # _150
    filter_threshold = 60  # bin2后
    out_channel = 2
elif type == "dend":
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/traindata/weights3/unet_150.pth"
    filter_threshold = 100
    out_channel = 1

maskpos_leftup_rightdown = []
# maskpos_leftup_rightdown = [[207,1434,250,1480],[117,0,150,30]]


################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#################
# 直接在本地base跑
# 调整版。好于predicttemp.py和predict.py
# 不用手动归一化输入,自动每套数据全局归一化，所以结果可能是与全局强度趋势相关的
# batchsize的调整


def sliding_window(image, window_size, stride, norm=False):
    patched = []
    _, h, w = image.shape
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            img = image[:, i : i + window_size, j : j + window_size]
            if norm:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img = img * 255

            patched.append(img)
            # patched.append(image[:,i:i+window_size,j:j+window_size])
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


def predict(data_path, weight_path, outchannel, filter_threshold, type):
    save_path = data_path + "/result"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print("Using device: ", device)
    torch.cuda.empty_cache()
    bin = 2
    # 3072x3072
    window_size = 256
    stride3072 = 192

    stride2048 = 128
    batch_size = 32  # 64
    threshold = 0.5  # 0.7

    norm_slice = False
    norm_patch = True

    out_channel = outchannel
    filter_threshold = filter_threshold
    # if weight_path.split('/')[-2][-1] == '2' or weight_path.split('/')[-2][-1] == '3':
    #     out_channel = 1
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
            input = tiff.imread(_input)
            if input.ndim == 2:
                input = np.expand_dims(input, axis=0)
            # print('input type',type(input))

            # #每套数据归一化0-255
            # input = np.ndarray.astype(input,np.float32)
            # input = (input - np.min(input)) / (np.max(input) - np.min(input))
            # inputs = input * 255

            # mask掉指定区域
            for pos in maskpos_leftup_rightdown:
                input[:, pos[1] : pos[3], pos[0] : pos[2]] = 0
            # input[:,maskpos_leftup_rightdown[1]:maskpos_leftup_rightdown[3],maskpos_leftup_rightdown[0]:maskpos_leftup_rightdown[2]] = 0
            print(max(input.flatten()))
            # 按99%分位数归一化
            input = np.ndarray.astype(input, np.float32)
            max_percentile = np.percentile(input, 99.95)
            print("max_percentile", max_percentile)
            input = (input - np.min(input)) / (max_percentile - np.min(input))
            inputs = input * 255
            inputs[inputs > 255] = 255

            results = np.zeros_like(inputs)
            # image = np.array(image)
            # image = image.astype(np.float32)
            # input = (input - np.min(input)) / (np.max(input) - np.min(input))
            # input = input * 255
            # print('input type',type(input))
            # print('input shape',input.size)
            if inputs.shape[1] == 2048:
                stride = stride2048
            else:
                stride = stride3072

            for t in range(inputs.shape[0]):
                print("t", t)
                inputst = inputs[t]
                if norm_slice:
                    inputst = (inputst - np.min(inputst)) / (
                        np.max(inputst) - np.min(inputst)
                    )
                    inputst = inputst * 255

                input = Image.fromarray(inputst)  # .convert('L')

                # #直方均横化(提前归一过滤最值)
                # input = ImageOps.equalize(input)

                input = transforms.Resize(
                    (int(input.size[0] / bin), int(input.size[1] / bin))
                )(input)
                print("input reshape", input.size)
                # reflect padding （window_size-stride）/2
                padded_input = transforms.Pad(
                    int((window_size - stride) / 2),
                    fill=0,
                    padding_mode="reflect",
                )(input)
                print("padded_input shape", padded_input.size)
                padded_input = np.reshape(
                    padded_input,
                    (1, padded_input.size[0], padded_input.size[1]),
                )
                padded_input = np.array(padded_input).astype("float32")
                # image = torch.from_numpy(padded_input)
                print("image shape", padded_input.shape)

                patches = sliding_window(
                    padded_input, window_size, stride, norm=norm_patch
                )
                print("patches shape", patches.shape)

                patches = torch.from_numpy(patches)
                patches = patches.to(device)
                print("patches shape", patches.shape)
                print(
                    "torch.cuda.memory_allocated():",
                    torch.cuda.memory_allocated(),
                )

                # for i in trange(0,int(patches.size()[0]/batch_size)):
                for i in trange(0, int(patches.size()[0] / batch_size)):
                    with torch.no_grad():
                        out = net(
                            patches[
                                i * batch_size : i * batch_size + batch_size
                            ]
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

                result[result > threshold] = 1
                result[result <= threshold] = 0
                print("filtering")
                result = filter_small(result, filter_threshold)

                print("result shape", result.shape)
                result = torch.from_numpy(result)
                result = result.float()
                print("tensor result shape", result.shape)
                result = torch.nn.functional.interpolate(
                    result.unsqueeze(0).unsqueeze(0),
                    size=(input.size[0] * bin, input.size[1] * bin),
                    mode="nearest",
                )
                print("tensor result reshape", result.shape)

                # result = np.resize(result,(result.shape[0]*bin,result.shape[1]*bin),)

                results[t] = result.detach().numpy()[0][0]

            # result = np.resize(result,(result.shape[0]*bin,result.shape[1]*bin),)

            # name = file[:-4]+'_'+str(weight_path.split('/')[-2])+'_'+str(weight_path.split('_')[-1][:-4]+'_result.tif')
            name = file[:-4] + "_" + type + "_result.tif"
            # print(name)
            # print('result shape',result.shape)
            # tifffile.imsave(os.path.join(save_path,name),result.detach().numpy()[0][0],dtype='uint8')
            # tifffile.imsave(os.path.join(save_path,name[:-4]+'_255.tif'),result255.detach().numpy()[0][0],dtype='uint8')
            print("saving")
            tifffile.imwrite(os.path.join(save_path, name), results)
            # tifffile.imwrite(os.path.join(save_path,name[:-4]+'_255.tif'),result255.detach().numpy()[0][0])


# for i in range(1,120):
#     weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train2/weights2/unet_'+str(i*5)+'.pth'
#     predict(data_path=data_path,weight_path=weight_path,filter_threshold=filter_threshold)
predict(
    data_path=data_path,
    weight_path=weight_path,
    outchannel=out_channel,
    filter_threshold=filter_threshold,
    type=type,
)
