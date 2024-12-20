import os

import numpy as np
import tifffile as tiff
from skimage import measure
from tqdm import trange

threshold = 50 * 2
datapath = "/share/data/CryoET_Data/lizhuo/pfatest/result1/3"


def filter_small(img, threshold):
    # 统计只有0,1二值的results中连续的1值区域，过滤小杂点
    result = np.array(img)
    print(result.shape)
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
        for j in trange(result.shape[0]):
            i = 1
            labeled_image, num_features = measure.label(
                result[j], connectivity=2, return_num=True
            )
            region_props = measure.regionprops(labeled_image)
            for props in region_props:
                if props.area < size_threshold:
                    labeled_image[labeled_image == props.label] = 0
                else:
                    labeled_image[labeled_image == props.label] = i
                    i += 1
            result[j] = labeled_image

    return result


for file in os.listdir(datapath):
    if file.endswith(".tif"):
        data = tiff.imread(os.path.join(datapath, file))
        data = filter_small(data, threshold)
        tiff.imwrite(os.path.join(datapath, file), data)
