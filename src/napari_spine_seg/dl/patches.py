import argparse
import os

import numpy as np
import tifffile as tiff
from scipy.ndimage import find_objects, label
from tqdm import trange

# 对文件夹下所有文件处理版

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="data folder path or file path")

args = parser.parse_args()

if args.path.endswith(".tif"):
    tif_files = [args.path.split("/")[-1]]
    data_folder = os.path.dirname(args.path)
else:
    data_folder = args.path
    tif_files = [f for f in os.listdir(data_folder) if f.endswith(".tif")]


# 设置文件夹路径
# data_folder = '/share/data/CryoET_Data/lizhuo/data_for_liushuo/3D_data_with_mask/l5'  # 文件夹路径
mask_signs = ("_spine_result", "-hwl-spine")
patch_size = 1024
overlap_rate = 0.5

# 获取文件夹中所有的 .tif 文件
# tif_files = [f for f in os.listdir(data_folder) if f.endswith('.tif')]


def equalize_histogram_sqrt(image):
    img_np = np.asarray(image)
    # 计算原始直方图
    histogram, bin_edges = np.histogram(
        img_np.flatten(), bins=256, range=(0, 256)
    )

    # 对直方图的值取平方根
    histogram_sqrt = np.sqrt(histogram)

    # 计算累计分布函数 (CDF)
    cdf = histogram_sqrt.cumsum()
    cdf_normalized = 255 * (cdf - cdf.min()) / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype("uint8")

    # 使用平方根处理后的 CDF 映射像素值
    equalized_img_np = cdf_normalized[img_np]

    return equalized_img_np


def image_preprocess(img, percentile=99.5, equalize=False):
    # input: PIL image or np array
    # max percentile + normalize -> 0-255 int(uint8)
    # output: np array
    if type(img) is np.ndarray:
        img_np = img
    else:
        img_np = np.array(img)

    max_percentile = np.percentile(img_np, percentile)
    img_np = np.minimum(img_np, max_percentile)
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
    img_np = img_np * 255
    img_np = img_np.astype(np.uint8)

    if equalize:
        img_np = equalize_histogram_sqrt(img_np)

    return img_np


def filter_small(img, threshold):
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


def reset_id(segmentation):
    # 获取唯一值
    values = np.unique(segmentation)

    # 检查是否需要重置 ID
    if np.max(segmentation) == len(values) - 1:
        return segmentation

    # 创建新的分割 ID 数组
    reset_id_seg = np.zeros_like(segmentation, dtype=int)
    # 过滤掉0值
    non_zero_values = values[values != 0]

    # 使用 np.arange 生成新 ID
    reset_id_seg[np.isin(segmentation, non_zero_values)] = np.arange(
        1, len(non_zero_values) + 1
    )[np.searchsorted(non_zero_values, segmentation[segmentation != 0])]

    return reset_id_seg.astype(np.uint16)


def get_patch(data, patch_size, overlap_rate):
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


# 遍历文件夹中的所有 .tif 文件
for tif_file in tif_files:
    print(f"Processing file: {tif_file}")

    data_path = os.path.join(data_folder, tif_file)
    file_name = tif_file.split(".")[0]
    save_path = os.path.join(data_folder, "patches")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if file_name.endswith(mask_signs):
        data_type = "mask"
    else:
        data_type = "data"

    if data_type == "data":
        data = tiff.imread(data_path)
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
        print(f"Processing data with shape: {data.shape}")
        processed_data = np.zeros(data.shape)
        for t in trange(data.shape[0]):
            processed_data[t] = image_preprocess(data[t])
        print("Getting patches for data...")
        data_patch = get_patch(processed_data, patch_size, overlap_rate)
        print("Saving data patches...")
        for i in trange(len(data_patch)):
            patch_file = os.path.join(save_path, f"{file_name}-{i:04d}.tif")
            # print("save_path:", save_path)
            # print('f"{file_name}-{i:04d}.tif":', f"{file_name}-{i:04d}.tif")
            # print("patch_file:", patch_file)
            tiff.imwrite(patch_file, data_patch[i].astype(np.uint8))

    if data_type == "mask":
        mask = tiff.imread(data_path)
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
        print(f"Reset id with mask shape: {mask.shape}")
        mask = filter_small(mask, 0)
        print("Getting patches for mask...")
        mask_patch = get_patch(mask, patch_size, overlap_rate)
        print("Saving mask patches...")
        for i in trange(len(mask_patch)):
            patch_file = os.path.join(save_path, f"{file_name}-{i:04d}.tif")
            tiff.imwrite(patch_file, mask_patch[i].astype(np.uint16))


print("All files processed!")
