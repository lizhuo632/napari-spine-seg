import numpy as np
import tifffile as tiff
from skimage import measure
from tqdm import trange

path = (
    "/share/data/CryoET_Data/lizhuo/SIM-data/mem-filo-train/161415_mask130.tif"
)
result = tiff.imread(path)
# 统计只有0,1二值的results中连续的1值区域，过滤小杂点
result = np.array(result)
size_threshold = 200

if len(result.shape) == 2:
    # 用measure.label函数将二值图像中的连通区域进行标记
    labeled_image, num_features = measure.label(
        result, connectivity=2, return_num=True
    )

    # 使用measure.regionprops函数获取每个连通区域的属性
    region_props = measure.regionprops(labeled_image)

    # 定义一个大小阈值，用来过滤小的杂点
    # 例如设定为100个像素

    # 遍历每个连通区域，根据面积大小进行过滤
    for props in region_props:
        if props.area < size_threshold:
            # 将面积小于阈值的区域对应的像素值设为0
            labeled_image[labeled_image == props.label] = 0

    result = np.where(labeled_image > 0, 1, 0)
else:
    for i in trange(result.shape[0]):
        labeled_image, num_features = measure.label(
            result[i], connectivity=2, return_num=True
        )
        region_props = measure.regionprops(labeled_image)
        for props in region_props:
            if props.area < size_threshold:
                labeled_image[labeled_image == props.label] = 0
        result[i] = np.where(labeled_image > 0, 1, 0)

tiff.imsave(path, result)
