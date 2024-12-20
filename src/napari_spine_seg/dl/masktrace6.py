from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import tifffile as tiff
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import trange

# 在集群的deeplearning环境下跑


def process_value(value, s0, s1, s2):
    if value != 0:
        mask = s2 == value
        count = Counter(s1.multiply(mask).data)
        if count.most_common(1) != []:
            max_overlap = count.most_common(1)[0][0]
            s2[mask] = max_overlap
        else:
            count = Counter(s0.multiply(mask).data)
            if count.most_common(1) != []:
                max_overlap = count.most_common(1)[0][0]
                s2[mask] = max_overlap
                s1[mask] = max_overlap
            # else:

        # overlapped_values = np.unique(s1.multiply(mask).data)
        # if len(overlapped_values) != 0:
        #     #s2[mask] = overlapped_values[0]
        #     max_overlap =
        # else:
        #     overlapped_values = np.unique(s0.multiply(mask).data)
        #     if len(overlapped_values) != 0:
        #         s2[mask] = overlapped_values[0]
        #         s1[mask] = overlapped_values[0]
        #     # else:
        #     global max_val
        #     #print("max_val:",max_val)
        #     max_val += 1
        #     s2[mask] = max_val


def process_value1(value, s0, s1, s2):
    if value != 0:
        mask = s2 == value
        count = Counter(s1.multiply(mask).data)
        if count.most_common(1) != []:
            max_overlap = count.most_common(1)[0][0]
            s2 = lil_matrix(s2)
            s2[mask] = max_overlap
            s2 = s2.tocsr()
        else:
            count = Counter(s0.multiply(mask).data)
            if count.most_common(1) != []:
                max_overlap = count.most_common(1)[0][0]
                s2 = lil_matrix(s2)
                s2[mask] = max_overlap
                s1[mask] = max_overlap
                s2 = s2.tocsr()


def slice_overlap(s0, s1, s2, threshold=0):
    s2_values = np.unique(s2.data)
    for value in s2_values:
        process_value(value, s0, s1, s2)


def renumber(data):
    data = data.astype(np.uint16)
    data_csr = sparse.csr_matrix(data.reshape(data.shape[0], -1))
    i = 1
    for num in np.unique(data_csr.data):
        mask = data_csr.data == num
        data_csr.data[mask] = i
        # print("num:",num,"i:",i)
        i += 1
    return data_csr.toarray().reshape(data.shape)


def renumber1(data):
    data = data.astype(np.uint16)
    data_csr = csr_matrix(data.reshape(data.shape[0], -1))
    unique_values = np.unique(data_csr.data)
    i = 1
    for num in unique_values:
        mask = data_csr.data == num
        data_lil = lil_matrix(data_csr.shape)
        data_lil.data[mask] = np.arange(i, i + np.sum(mask))
        data_csr = data_lil.tocsr()
        i += np.sum(mask)
    return data_csr.toarray().reshape(data.shape)


def data_overlap(datapath, bin=1):
    data = tiff.imread(datapath)
    result = data.copy()
    # bin
    data = data.astype(np.uint16)
    bin = 1  # 4
    databin = data[:, ::bin, ::bin]
    data = databin
    # global max_val
    max_val = 0  # int(np.max(data))
    # data_csr = sparse.csr_matrix(data.reshape(data.shape[0], -1))

    for i in trange(data.shape[0] - 1):

        data_i_csr = sparse.csr_matrix(data[i].reshape(1, -1))
        max_val = int(np.max(data_i_csr.data))
        if i == 0:
            data_isub1_csr = data_i_csr.copy()
        else:
            data_isub1_csr = sparse.csr_matrix(data[i - 1].reshape(1, -1))
        data_iplus1_csr = sparse.csr_matrix(data[i + 1].reshape(1, -1))
        data_iplus1_csr.data += max_val
        slice_overlap(data_isub1_csr, data_i_csr, data_iplus1_csr)
        np.copyto(
            data[i + 1], data_iplus1_csr.toarray().reshape(data.shape[1:])
        )

    renumeddata = renumber(data)
    np.copyto(data, renumeddata)
    print("max_val:", np.max(data))
    tiff.imwrite(datapath.split(".")[-2] + "_overlap.tif", data)
    print("save overlap data to:", datapath.split(".")[-2] + "_overlap.tif")


def exist_t(datapath):
    data = tiff.imread(datapath)
    data = data.astype(np.uint16)
    max_val = int(np.max(data))
    max_t = data.shape[0]
    exist = np.zeros(max_val + 1)

    unique_m = [np.unique(data[i]) for i in range(data.shape[0])]
    for i in trange(data.shape[0]):
        exist[unique_m[i]] += 1

    # save exist
    with open(datapath.split(".")[0] + "_exist.txt", "w") as f:
        for i in range(len(exist)):
            f.write(str(int(exist[i])) + "\n")

    fig = plt.figure(figsize=(30, 10))
    plt.hist(exist, bins=range(1, max_t + 1))
    plt.xticks(range(1, max_t + 1))
    plt.tick_params(labelsize=5)
    h, e = np.histogram(exist, bins=range(1, max_t + 1))
    for i, h in enumerate(h):
        plt.text(
            (e[i] + e[i + 1]) / 2,
            h + 5,
            str(h),
            horizontalalignment="center",
            fontsize=8,
        )

    plt.savefig(datapath.split(".")[0] + "_exist_hist.png")


def delete_small_t(datapath):
    data = tiff.imread(datapath)
    with open(datapath.split(".")[0] + "_exist.txt") as f:
        exist = f.readlines()
    exists = [int(i) for i in exist]
    for t in range(data.shape[0]):
        vals_in_t = np.unique(data[t])
        for val in vals_in_t:
            if exists[val] <= 5:
                data[t][data[t] == val] = 0
    tiff.imwrite(datapath.split(".")[0] + "_delete_small_t.tif", data)
    print(
        "save delete_small_t data to:",
        datapath.split(".")[0] + "_delete_small_t.tif",
    )

    exist_t(datapath.split(".")[0] + "_delete_small_t.tif")


def sum_size(data_path, mask_path, psd_mask_path):
    data = tiff.imread(datapath)
    mask = tiff.imread(mask_path)
    psd_mask = tiff.imread(psd_mask_path)
    intensities = []
    sizes = []
    punctas_num = []
    punctas_size = []
    for i in range(data.shape[0]):
        intensities.append(np.mean(data[i, mask[i] >= 1]))
        sizes.append(np.sum(mask[i] >= 1))
        punctas_num.append(np.unique(psd_mask[i][mask[i] >= 1]).shape[0])
        punctas_size.append(np.sum(psd_mask[i][mask[i] >= 1] >= 1))
    # print(intensities)
    fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    ax[0].plot(intensities)
    ax[0].set_title("intensities")
    ax[1].plot(sizes)
    ax[1].set_title("sizes")
    ax[2].plot(punctas_num)
    ax[2].set_title("punctas_num")
    ax[3].plot(punctas_size)
    ax[3].set_title("punctas_size")
    plt.tight_layout()
    plt.savefig(mask_path.split(".")[0] + "_total_analysis.png")


def spine_analysis(id, img, mask, psd_mask):
    # img = tiff.imread(imgpath)
    # mask = tiff.imread(datapath)
    # psd_mask = tiff.imread(psd_mask_path)
    # id = 0
    sizes = []
    intensities = []
    punctas_num = []
    punctas_size = []

    for i in range(img.shape[0]):
        spine_1 = mask[i] == id
        spine_size = np.sum(spine_1)
        spine_intensity = np.mean(img[i, spine_1] + 0)
        puncta_num = np.unique(psd_mask[i][spine_1]).shape[0]
        puncta_size = np.sum(psd_mask[i][spine_1] >= 1)
        sizes.append(spine_size)
        intensities.append(np.nan_to_num(spine_intensity))
        punctas_num.append(np.nan_to_num(puncta_num))
        punctas_size.append(puncta_size)

    return sizes, intensities, punctas_num, punctas_size


def spines_analysis(imgpath, datapath, psd_mask_path, exist_t_path):
    img = tiff.imread(imgpath)
    mask = tiff.imread(datapath)
    psd_mask = tiff.imread(psd_mask_path)
    exist_t = np.loadtxt(exist_t_path)
    with open(datapath.split(".")[0] + "_exist.txt") as f:
        exist = f.readlines()
    exists = [int(i) for i in exist]
    fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    sum_sizes = np.zeros(img.shape[0])
    sum_intensities = np.zeros(img.shape[0])
    sum_punctas_num = np.zeros(img.shape[0])
    sum_punctas_size = np.zeros(img.shape[0])
    spine_num = 0
    for id in range(1, exist_t.shape[0]):
        if exists[id] > 80:  # 80:
            print("id:", id)
            sizes, intensities, punctas_num, punctas_size = spine_analysis(
                id, img, mask, psd_mask
            )
            sum_sizes += np.array(sizes)
            sum_intensities += np.array(intensities)
            sum_punctas_num += np.array(punctas_num)
            sum_punctas_size += np.array(punctas_size)
            spine_num += 1

            ax[0].plot(intensities)
            ax[0].set_title("intensities")
            ax[1].plot(sizes)
            ax[1].set_title("sizes")
            ax[2].plot(punctas_num)
            ax[2].set_title("punctas_num")
            ax[3].plot(punctas_size)
            ax[3].set_title("punctas_size")
    plt.tight_layout()
    plt.savefig(datapath.split(".")[0] + "_spines_analysis.png")
    plt.close()
    fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    ax[0].plot(sum_intensities / spine_num)
    ax[0].set_title("avg_intensities")
    ax[1].plot(sum_sizes / spine_num)
    ax[1].set_title("avg_sizes")
    ax[2].plot(sum_punctas_num / spine_num)
    ax[2].set_title("avg_punctas_num")
    ax[3].plot(sum_punctas_size / spine_num)
    ax[3].set_title("avg_punctas_size")
    plt.tight_layout()
    plt.savefig(datapath.split(".")[0] + "_spines_avg_analysis.png")
    plt.close()


def show_good_spine(datapath, exist_t_path):
    mask = tiff.imread(datapath)
    exist_t = np.loadtxt(exist_t_path)
    with open(datapath.split(".")[0] + "_exist.txt") as f:
        exist = f.readlines()
    exists = [int(i) for i in exist]
    mask_good = np.zeros(mask.shape)
    for id in range(1, exist_t.shape[0]):
        print("id:", id)
        if exists[id] > 80:  # > 80:
            mask_good[mask == id] = 1

    tiff.imwrite(datapath.split(".")[0] + "_good.tif", mask_good)


if __name__ == "__main__":

    datapath = "/share/data/CryoET_Data/lizhuo/test/sam-trace/#Stablized AVG_roi2_seq3_3D-SIM561_RedCh_SIrecon_actin_result.tif"
    data_overlap(datapath)
    exit()

    imgpath = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/0423data/Stablized AVG_roi1_seq3_3D-SIM561_RedCh_SIrecon.tif"
    datapath = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/0423data/result2/Stablized AVG_roi1_seq3_3D-SIM561_RedCh_SIrecon_actin_result.tif"
    psd_mask_path = "/share/data/CryoET_Data/lizhuo/SIM-data/240118/2/2_640_psd_mask.tif"  # tif
    # imgpath = '/share/data/CryoET_Data/lizhuo/wenlan0307/AVG_seq0_StackedSlices-SIM561_RedCh_SIrecon-gauss.tif' #tif
    # datapath = '/share/data/CryoET_Data/lizhuo/wenlan0307/result1/AVG_seq0_StackedSlices-SIM561_RedCh_SIrecon-gauss_actin_result.tif' #tif
    # psd_mask_path = '/share/data/CryoET_Data/lizhuo/SIM-data/240118/2/2_640_psd_mask.tif' #tif

    # imgpath = '/share/data/CryoET_Data/lizhuo/work_with_wenlan/cLTP_data/ylq_cLTP-20min-5min-1h_DATA/3/ctrl3_561.tif' #tif
    # datapath = '/share/data/CryoET_Data/lizhuo/work_with_wenlan/cLTP_data/ylq_cLTP-20min-5min-1h_DATA/3/result1/ctrl3_561_actin_result.tif' #tif
    # psd_mask_path = '/share/data/CryoET_Data/lizhuo/work_with_wenlan/cLTP_data/ylq_cLTP-20min-5min-1h_DATA/3/result1/ctrl3_640_psd_mask.tif' #tif

    data_overlap(datapath)
    data_overlap_path = datapath.split(".")[-2] + "_overlap.tif"
    exist_t(datapath=data_overlap_path)
    delete_small_t(
        datapath=data_overlap_path
    )  # 过滤太强了？还是得分类讨论，拼接。

    # 过滤后所有spine的统计
    sum_size(
        data_path=imgpath,
        mask_path=data_overlap_path.split(".")[0] + "_delete_small_t.tif",
        psd_mask_path=psd_mask_path,
    )

    # 过滤后存在大于80帧spine的统计（平均值，与 全部趋势重叠画在一张图上）
    spines_analysis(
        imgpath=imgpath,
        datapath=data_overlap_path.split(".")[0] + "_delete_small_t.tif",
        psd_mask_path=psd_mask_path,
        exist_t_path=data_overlap_path.split(".")[0]
        + "_delete_small_t_exist.txt",
    )

    show_good_spine(
        datapath=data_overlap_path.split(".")[0] + "_delete_small_t.tif",
        exist_t_path=data_overlap_path.split(".")[0]
        + "_delete_small_t_exist.txt",
    )
