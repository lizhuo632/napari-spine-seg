# condadl上CUDA运行,或本地base用CPU运行,适配napari插件的版本

# from skimage import measure
import os

import albumentations as A
import numpy as np
import tifffile as tiff
import torch
from scipy.ndimage import find_objects, label
from scipy.ndimage.filters import gaussian_filter
from tqdm import trange

from .datapreprocess import *
from .net import *
from .utils import image_preprocess

# device = torch.device("cpu")

# print('torch.cuda.memory_allocated():',torch.cuda.memory_allocated())


type = "dend"
type = "memspine"  #'mem'
# type = 'psd'


type = "spine"

type = "presyn"


# data_path = '/share/data/CryoET_Data/lizhuo/wenlan_data/alpha-syn/predict/0725-parts'#'/share/data/CryoET_Data/lizhuo/SIM-data/240118/2/predict'#'/share/data/CryoET_Data/lizhuo/test/predicttest/class'#'/share/data/CryoET_Data/lizhuo/preprocessed_data/synap488_actin561_psd640/#20231201_1113culture_Mg-ctrl_cLTP-coverslip2/roi2/test2'#'/share/data/CryoET_Data/lizhuo/work_with_wenlan/0423data/singleframetest' #wenlan0307/test6' #/SIM-data/240118/2_norm # data/
# data_path = '/share/data/CryoET_Data/lizhuo/work_with_wenlan/cLTP_data/ylq_cLTP-20min-5min-1h_DATA/5'
# data_path = '/share/data/CryoET_Data/lizhuo/wenlan_data/alpha-syn/predict/tests/488'
data_path = "/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/syn_actin561-data analysis-20240802_20240716culture"
data_path = "/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/syn_actin561-data analysis-20240802_20240716culture/33"
data_path = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2-predict-results"
data_path = (
    "/share/data/CryoET_Data/lizhuo/data_for_liushuo/3D_training_data/mask2"
)
data_path = (
    "/share/data/CryoET_Data/lizhuo/SIM-process/241204/roi1-ttx-patch/test"
)
data_path = (
    "/share/data/CryoET_Data/lizhuo/SIM-process/241204/roi1-ttx-patch/presyn"
)

dropout = False
logits = True
if type == "spine":
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train1/weights1/unet_400.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train3/weights1/unet_134.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train4/weights1/unet_100.pth"  ###比较好用
    # weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train10/weights2/unet_202.pth'
    # weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train10/weights2/unet_109.pth'
    # weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train5/weights1/unet_100.pth'
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train11_wldata2/weights1/unet_202.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/weights1/unet_115.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/weights2/unet_180.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13/weights1/unet_29.pth"
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13/weights2/unet_34.pth"

    # weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/weights111/unet_114.pth'

    logits = False
    logits = True
    filter_threshold = 80  # 50#25 #bin2后
    out_channel = 2
    temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
    threshold = 0.1
if type == "presyn":
    # 用spine的网络
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/weights2/unet_180.pth"

    logits = False
    logits = True
    filter_threshold = 80  # 50#25 #bin2后
    out_channel = 2
    temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
    threshold = 0.5
if type == "psd":
    # 用actin的网络,效果不好，待训或直接用imagej的binary-convert to mask
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/weights2/unet_5.pth"

    logits = False
    logits = True
    filter_threshold = 10  # 50#25 #bin2后
    out_channel = 2
    temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
    threshold = 0.5


elif type == "memspine":
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo/weights0/weig_200.pth"  # _150
    filter_threshold = 60  # bin2后
    out_channel = 2
    temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
    threshold = 0.5
elif type == "dend":
    weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/traindata/weights3/unet_150.pth"  # 150
    filter_threshold = 500  # 100
    out_channel = 1
    logits = False
    dropout = True
    temp_rate = 1  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
    threshold = 0.1
maskpos_leftup_rightdown = []
# maskpos_leftup_rightdown = [[207,1434,250,1480],[117,0,150,30]]


def filter_small2(img, threshold):
    result = np.array(img)
    if len(result.shape) == 2:
        # 使用 SciPy 的 label 函数标记连通区域
        labeled_image, num_features = label(result)

        # 获取每个区域的边界框
        objects = find_objects(labeled_image)

        for i, obj in enumerate(objects, start=1):
            area = np.sum(labeled_image[obj] == i)
            if area < threshold:
                labeled_image[labeled_image == i] = 0
            else:
                labeled_image[labeled_image == i] = i

        result = labeled_image
    else:
        for t in range(result.shape[0]):
            labeled_image, num_features = label(result[t])
            objects = find_objects(labeled_image)
            for i, obj in enumerate(objects, start=1):
                area = np.sum(labeled_image[obj] == i)
                if area < threshold:
                    labeled_image[labeled_image == i] = 0
                else:
                    labeled_image[labeled_image == i] = i
            result[t] = labeled_image

    return result


def get_gaussian(s, sigma=6.0 / 8) -> np.ndarray:
    temp = np.zeros(s)
    coords = [i // 2 for i in s]
    sigmas = [i * sigma for i in s]
    temp[tuple(coords)] = 1
    gaussian_map = gaussian_filter(temp, sigmas, 0, mode="constant", cval=0)
    gaussian_map /= np.max(gaussian_map)
    return gaussian_map.astype(np.float32)


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


def get_patch(data, patch_size, overlap_rate):
    """
    切割图像为patch，重叠率为overlap_rate
    """
    data_patch = []
    for i in range(
        0,
        data.shape[1] - int(patch_size * (1 - overlap_rate)),
        int(patch_size * (1 - overlap_rate)),
    ):
        for j in range(
            0,
            data.shape[2] - int(patch_size * (1 - overlap_rate)),
            int(patch_size * (1 - overlap_rate)),
        ):
            data_patch.append(data[:, i : i + patch_size, j : j + patch_size])
    return data_patch


def predict0(
    data_path,
    weight_path,
    outchannel,
    threshold,
    filter_threshold,
    type,
    logits=True,
    dropout=False,
    temp_rate=255,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    # device = torch.device("cpu")
    print("Using device: ", device)
    torch.cuda.empty_cache()
    bin = 2
    # 3072x3072
    patch_size = 256
    gussian_map = get_gaussian((patch_size, patch_size))
    stride3072 = 192

    stride2048 = 128
    stride1024 = 64
    batch_size = 32  # 64
    # threshold = 0.1#3#5 # 0.7
    save_path = (
        data_path
        + "/0result-"
        + weight_path.split("/")[-3]
        + "-"
        + weight_path.split("/")[-2]
        + "-"
        + weight_path.split("/")[-1][:-4]
        + "-th_0"
        + str(threshold).split(".")[1]
        + "-ft_"
        + str(filter_threshold)
        + "-"
        + type
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    norm_slice = False
    norm_patch = False

    out_channel = outchannel
    # print('out_channel: ',out_channel)
    filter_threshold = filter_threshold
    # if weight_path.split('/')[-2][-1] == '2' or weight_path.split('/')[-2][-1] == '3':
    #     out_channel = 1
    net = UNet(
        in_channels=1, out_channels=out_channel, logits=logits, dropout=dropout
    ).to(device)

    # net = ResUNet().to(device)
    net.load_state_dict(
        torch.load(weight_path, map_location=torch.device(device))
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

            # mask掉指定区域
            for pos in maskpos_leftup_rightdown:
                input[:, pos[1] : pos[3], pos[0] : pos[2]] = 0
            # input[:,maskpos_leftup_rightdown[1]:maskpos_leftup_rightdown[3],maskpos_leftup_rightdown[0]:maskpos_leftup_rightdown[2]] = 0

            inputs = input  # utils.image_preprocess(input, percentile=99.5, equalize=False) 不在这做，在每个slice做

            results = np.zeros_like(inputs).astype("uint16")

            for t in range(inputs.shape[0]):
                print("t", t)
                inputst = inputs[t]
                input = image_preprocess(
                    inputst, percentile=99.5, equalize=False
                )

                input = A.Resize(
                    int(input.shape[0] / bin), int(input.shape[1] / bin)
                )(image=input)["image"]
                print("input shape", input.shape)

                # Padding to match window size
                pad_size = 128
                padded_input = np.pad(
                    input,
                    ((pad_size, pad_size), (pad_size, pad_size)),
                    "reflect",
                )
                print("padded image shape", padded_input.shape)

                # Reshape padded input for processing

                # Extract patches with overlap
                overlap_rate = 0.5
                patches = get_patch(
                    np.reshape(
                        padded_input,
                        (1, padded_input.shape[0], padded_input.shape[1]),
                    ),
                    patch_size,
                    overlap_rate=overlap_rate,
                )
                patch_num = int(np.ceil(np.sqrt(len(patches))))
                stride = int(patch_size * (1 - overlap_rate))
                print("number of patches:", len(patches))

                patches = torch.from_numpy(
                    np.array(patches).astype("float32") / temp_rate
                )
                patches = patches.to(device)

                # Process patches in batches
                for i in trange(0, int(patches.size()[0] / batch_size) + 1):
                    with torch.no_grad():
                        out = net(
                            patches[
                                i * batch_size : i * batch_size + batch_size
                            ]
                        )
                        if logits:
                            out_prob = torch.nn.functional.softmax(out, dim=1)
                            out = out_prob

                    if i == 0:
                        outs = out
                    else:
                        outs = torch.cat((outs, out), dim=0)

                print("outs shape", outs.shape)

                # Create result image (padded)
                result_padded = np.zeros_like(padded_input).astype("float32")
                normalization = np.zeros_like(padded_input).astype("float32")
                # tiff.imwrite(os.path.join(save_path,'out0.tif'),outs[0][out_channel - 1].cpu().detach().numpy().astype('float32'))
                # for i in range(0, outs.shape[0]):
                for row in trange(patch_num):
                    for col in range(patch_num):
                        # row = int(i / (int(padded_input.shape[0] / stride)))
                        # col = int(i % (int(padded_input.shape[1] / stride)))
                        i = row * patch_num + col
                        _out = outs[i][out_channel - 1].cpu().detach().numpy()
                        _out = _out * gussian_map

                        # print('row,col:',row,col)

                        result_padded[
                            row * stride : row * stride + patch_size,
                            col * stride : col * stride + patch_size,
                        ] += _out.astype("float32")
                        normalization[
                            row * stride : row * stride + patch_size,
                            col * stride : col * stride + patch_size,
                        ] += gussian_map

                result_padded_norm = result_padded / normalization

                # Remove padding to restore original image size
                result = result_padded_norm[
                    pad_size:-pad_size, pad_size:-pad_size
                ]

                result = torch.from_numpy(result)
                result = result.float()

                result = torch.nn.functional.interpolate(
                    result.unsqueeze(0).unsqueeze(0),
                    size=(input.shape[0] * bin, input.shape[1] * bin),
                    mode="nearest",
                )
                # tiff.imwrite(os.path.join(save_path,'raw.tif'),result.detach().numpy()[0][0].astype('float32'))
                result[result > threshold] = 1
                result[result <= threshold] = 0

                print("filtering")
                result = filter_small2(
                    result.detach().numpy()[0][0], filter_threshold
                )
                results[t] = result

            name = file[:-4] + "_" + type + "_result.tif"
            print("saving")
            tiff.imwrite(
                os.path.join(save_path, name), results.astype("uint16")
            )


# for i in range(1,120):
#     weight_path = '/share/data/CryoET_Data/lizhuo/SIM-data/actin/train2/weights2/unet_'+str(i*5)+'.pth'
#     predict(data_path=data_path,weight_path=weight_path,filter_threshold=filter_threshold)
# predict(data_path=data_path,weight_path=weight_path,outchannel=out_channel,threshold=threshold, filter_threshold=filter_threshold,type=type,logits=logits,dropout=dropout,temp_rate=temp_rate)


def predict(data, type, target, device, normlize=False):
    device = torch.device(device)
    dropout = False
    logits = True
    if type == "actin" and target == "spine":
        weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13/weights2/unet_34.pth"
        logits = True
        filter_threshold = 80  # 50#25 #bin2后
        out_channel = 2
        temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
        threshold = 0.1
    if target == "pre-synapse":
        # 用spine的网络
        weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/weights2/unet_180.pth"
        logits = True
        filter_threshold = 80  # 50#25 #bin2后
        out_channel = 2
        temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
        threshold = 0.5
    if target == "psd":
        # 用actin的网络,效果不好，待训或直接用imagej的binary-convert to mask
        weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train12/weights2/unet_5.pth"
        logits = False
        logits = True
        filter_threshold = 10  # 50#25 #bin2后
        out_channel = 2
        temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
        threshold = 0.5
    elif type == "myr" and target == "spine":
        weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/mem/mem-spine+filo/weights0/weig_200.pth"  # _150
        filter_threshold = 60  # bin2后
        out_channel = 2
        temp_rate = 255  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
        threshold = 0.5
    elif type == "actin" and target == "dend":
        weight_path = "/share/data/CryoET_Data/lizhuo/SIM-data/traindata/weights3/unet_150.pth"  # 150
        filter_threshold = 500  # 100
        out_channel = 1
        logits = False
        dropout = True
        temp_rate = 1  # image preprocess后为0-255,/255归一输入网络。但有些网络训练时/65535.待重训。
        threshold = 0.1

    torch.cuda.empty_cache()
    bin = 2
    # 3072x3072
    patch_size = 256
    gussian_map = get_gaussian((patch_size, patch_size))

    batch_size = 32  # 64
    # threshold = 0.1#3#5 # 0.7
    # save_path = data_path+'/0result-'+weight_path.split('/')[-3]+'-'+weight_path.split('/')[-2]+'-'+weight_path.split('/')[-1][:-4]+'-th_0'+str(threshold).split('.')[1]+'-ft_'+str(filter_threshold)+'-'+type
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    norm_slice = False
    norm_patch = False

    # print('out_channel: ',out_channel)
    filter_threshold = filter_threshold
    # if weight_path.split('/')[-2][-1] == '2' or weight_path.split('/')[-2][-1] == '3':
    #     out_channel = 1
    net = UNet(
        in_channels=1, out_channels=out_channel, logits=logits, dropout=dropout
    ).to(device)

    # net = ResUNet().to(device)
    net.load_state_dict(
        torch.load(weight_path, map_location=torch.device(device))
    )
    net.eval()
    print("load: ", weight_path)

    input = data
    if input.ndim == 2:
        input = np.expand_dims(input, axis=0)
    # print('input type',type(input))

    # mask掉指定区域
    for pos in maskpos_leftup_rightdown:
        input[:, pos[1] : pos[3], pos[0] : pos[2]] = 0
    # input[:,maskpos_leftup_rightdown[1]:maskpos_leftup_rightdown[3],maskpos_leftup_rightdown[0]:maskpos_leftup_rightdown[2]] = 0

    inputs = input  # utils.image_preprocess(input, percentile=99.5, equalize=False) 不在这做，在每个slice做

    results = np.zeros_like(inputs).astype("uint16")

    for t in range(inputs.shape[0]):
        print("t", t)
        inputst = inputs[t]
        if normlize:
            input = image_preprocess(inputst, percentile=99.5, equalize=False)
        else:
            input = inputst

        input = A.Resize(int(input.shape[0] / bin), int(input.shape[1] / bin))(
            image=input
        )["image"]
        print("input shape", input.shape)

        # Padding to match window size
        pad_size = 128
        padded_input = np.pad(
            input, ((pad_size, pad_size), (pad_size, pad_size)), "reflect"
        )
        print("padded image shape", padded_input.shape)

        # Reshape padded input for processing

        # Extract patches with overlap
        overlap_rate = 0.5
        patches = get_patch(
            np.reshape(
                padded_input, (1, padded_input.shape[0], padded_input.shape[1])
            ),
            patch_size,
            overlap_rate=overlap_rate,
        )
        patch_num = int(np.ceil(np.sqrt(len(patches))))
        stride = int(patch_size * (1 - overlap_rate))
        print("number of patches:", len(patches))

        patches = torch.from_numpy(
            np.array(patches).astype("float32") / temp_rate
        )
        patches = patches.to(device)

        # Process patches in batches
        for i in trange(0, int(patches.size()[0] / batch_size) + 1):
            with torch.no_grad():
                out = net(
                    patches[i * batch_size : i * batch_size + batch_size]
                )
                if logits:
                    out_prob = torch.nn.functional.softmax(out, dim=1)
                    out = out_prob

            if i == 0:
                outs = out
            else:
                outs = torch.cat((outs, out), dim=0)

        print("outs shape", outs.shape)

        # Create result image (padded)
        result_padded = np.zeros_like(padded_input).astype("float32")
        normalization = np.zeros_like(padded_input).astype("float32")
        # tiff.imwrite(os.path.join(save_path,'out0.tif'),outs[0][out_channel - 1].cpu().detach().numpy().astype('float32'))
        # for i in range(0, outs.shape[0]):
        for row in trange(patch_num):
            for col in range(patch_num):
                # row = int(i / (int(padded_input.shape[0] / stride)))
                # col = int(i % (int(padded_input.shape[1] / stride)))
                i = row * patch_num + col
                _out = outs[i][out_channel - 1].cpu().detach().numpy()
                _out = _out * gussian_map

                # print('row,col:',row,col)

                result_padded[
                    row * stride : row * stride + patch_size,
                    col * stride : col * stride + patch_size,
                ] += _out.astype("float32")
                normalization[
                    row * stride : row * stride + patch_size,
                    col * stride : col * stride + patch_size,
                ] += gussian_map

        result_padded_norm = result_padded / normalization

        # Remove padding to restore original image size
        result = result_padded_norm[pad_size:-pad_size, pad_size:-pad_size]

        result = torch.from_numpy(result)
        result = result.float()

        result = torch.nn.functional.interpolate(
            result.unsqueeze(0).unsqueeze(0),
            size=(input.shape[0] * bin, input.shape[1] * bin),
            mode="nearest",
        )
        # tiff.imwrite(os.path.join(save_path,'raw.tif'),result.detach().numpy()[0][0].astype('float32'))
        result[result > threshold] = 1
        result[result <= threshold] = 0

        print("filtering")
        result = filter_small2(result.detach().numpy()[0][0], filter_threshold)
        results[t] = result

    return results
