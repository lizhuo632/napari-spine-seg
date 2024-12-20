import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import trange


class SIM:

    def __init__(self, datapath, maskpath, psdmaskpath):

        try:
            self.data = np.load(
                datapath, allow_pickle=True
            )  # self.data.shape = (t,x,y)
        except:
            data = tifffile.imread(datapath)  # 输入如果是tif
            output_file = os.path.splitext(datapath)[0] + ".npy"
            np.save(output_file, data)
            self.data = np.load(
                output_file, allow_pickle=True
            )  # self.data.shape = (t,x,y)
            datapath = output_file

        try:
            self.mask = np.load(
                maskpath, allow_pickle=True
            )  # self.mask.shape = (t,x,y)
        except:
            mask = tifffile.imread(maskpath)
            output_file = os.path.splitext(maskpath)[0] + ".npy"
            np.save(output_file, mask)
            self.mask = np.load(output_file, allow_pickle=True)
            maskpath = output_file

        try:
            self.psdmask = np.load(
                psdmaskpath, allow_pickle=True
            )  # self.mask.shape = (t,x,y)
        except:
            psdmask = tifffile.imread(psdmaskpath)
            output_file = os.path.splitext(psdmaskpath)[0] + ".npy"
            np.save(output_file, psdmask)
            self.psdmask = np.load(output_file, allow_pickle=True)
            psdmaskpath = output_file

        self.datapath = datapath
        self.maskpath = maskpath

    def size(self, spine_number):
        sizes = []
        for i in range(self.mask.shape[0]):
            if np.count_nonzero(np.where(self.mask[i] == spine_number)) == 0:
                sizes.append(0)
            else:
                sizes.append(
                    np.nan_to_num(
                        np.count_nonzero(
                            np.where(self.mask[i] == spine_number)
                        ),
                        nan=0,
                    )
                )
        return sizes

    def position(self, spine_number):
        positions = []
        for i in range(self.mask.shape[0]):
            if np.count_nonzero(np.where(self.mask[i] == spine_number)) == 0:
                positions.append([0, 0])
            else:
                positions.append(
                    np.nan_to_num(
                        np.median(
                            np.where(self.mask[i] == spine_number), axis=1
                        ),
                        nan=0,
                    )
                )

        return np.array(positions).T.astype(int)

    # def exist(self):
    #     exist = []
    #     with open('/share/data/CryoET_Data/lizhuo/SIM-process/1/exist.txt','w') as f:
    #         for i in trange(1,np.max(self.mask)+1):
    #             t = 0
    #             #print(i)
    #             for j in range(self.mask.shape[0]):
    #                 if np.count_nonzero(np.where(self.mask[j] == i)) != 0:
    #                     t += 1
    #             #print(t)
    #             f.write(str(t)+'\n')
    #             exist.append(t)
    #     return exist
    def exist(self):
        max = np.max(self.mask)
        exist = np.zeros(max + 1)
        for i in trange(self.mask.shape[0]):
            exist[np.unique(self.mask[i])] += 1
        with open(os.path.splitext(self.datapath)[0] + "_exist.txt", "w") as f:
            for i in range(max + 1):
                f.write(str(int(exist[i])) + "\n")
        return exist

    def shape():
        pass

    def intensity(self, spine_number):
        intensity = []
        for i in range(self.mask.shape[0]):
            if np.count_nonzero(np.where(self.mask[i] == spine_number)) == 0:
                intensity.append(0.0)
            else:
                intensity.append(
                    np.nan_to_num(
                        np.nanmean(
                            self.data[i][
                                np.where(self.mask[i] == spine_number)
                            ]
                        ),
                        nan=0,
                    )
                )
        return intensity

    def puncta(self, spine_number):
        # 检查spine对应的psdmask,返回每一帧的puncta数目、总大小
        puncta_num = []
        puncta_size = []
        for i in range(self.psdmask.shape[0]):
            # 01二值化psdmask
            psdmask_i = np.zeros_like(self.psdmask[i])
            psdmask_i[self.psdmask[i] > 0] = 1
            spine_i = np.zeros_like(self.mask[i])
            spine_i[self.mask[i] == spine_number] = 1
            # 将spine与psdmask相乘，得到重叠的部分
            overlap = np.multiply(self.mask[i], psdmask_i)
            puncta_num.append(
                np.count_nonzero(
                    np.unique(np.multiply(spine_i, self.psdmask[i]))
                )
            )
            puncta_size.append(
                np.count_nonzero(overlap[overlap == spine_number])
            )
        return np.array(puncta_num), np.array(puncta_size)

    def summary_data(self):
        print("Data: %s" % self.datapath)
        print("Mask: %s" % self.maskpath)
        print("t: %d" % self.data.shape[0])
        print("Number of spines: %d" % np.max(self.mask))
        #       print('Number of punctas: %d' % np.max(self.puncta))
        #       print('Number of dendrites: %d' % np.max(self.dendrite)) 可以很强地滤波一下吗？

        # exist = []
        # for i in range(1,np.max(self.mask)+1):
        #     exist.append(np.count_nonzero(self.size(i)))
        #     if i%1== 0:
        #         print(i)
        # print(exist[1])
        # plt.hist(exist,bins=range(1,np.max(self.mask)+2))
        # plt.title('Spine Existence')
        # plt.show()

        exist = self.exist()
        print(exist[1])
        plt.hist(exist, bins=range(1, np.max(self.mask) + 2))
        plt.title("Spine Existence")
        plt.show()

    def summary_spine(self, spine_number):

        size = self.size(spine_number)
        intensity = self.intensity(spine_number)
        positionx, positiony = self.position(spine_number)
        t = len(size)
        firstt = np.nonzero(size)[0][0]

        print("Spine ID: %d" % spine_number)
        print("Spine existed in %d of %d frames" % (np.count_nonzero(size), t))
        print("Spine Average Size: %.0f" % np.mean(size[size != 0]))
        print(
            "Spine Position:(%.0f,%.0f)"
            % (
                np.mean(positionx[positionx != 0]),
                np.mean(positiony[positiony != 0]),
            )
        )
        print(
            "Spine Average Intensity: %f" % np.mean(intensity[intensity != 0])
        )
        puncta_num, puncta_size = self.puncta(spine_number)
        print("Punctas Number of spine:", puncta_num)
        print("Punctas Size of spine:", puncta_size)

        fig, ax = plt.subplots(3, 2)
        ax[0, 0].plot(range(self.data.shape[0]), size)
        ax[0, 0].set_title("Spine Size")
        ax[0, 0].set_xlabel("t")
        ax[0, 0].set_ylabel("Size: pixel")
        # ax[0,0].set_xticks(range(1,t+1))
        ax[0, 1].plot(range(self.data.shape[0]), intensity)
        ax[0, 1].set_title("Spine Intensity")
        ax[0, 1].set_xlabel("t")
        ax[0, 1].set_ylabel("Intensity: ")
        # ax[0,1].set_xticks(range(1,t+1))
        ax[1, 0].plot(positionx[positionx != 0], positiony[positiony != 0])
        ax[1, 0].set_title("Spine Position")
        ax[1, 0].set_xlabel("x")
        ax[1, 0].set_ylabel("y")
        # 坐标轴范围在+-25之间
        ax[1, 0].set_xlim(
            [
                np.mean(positionx[positionx != 0]) - 25,
                np.mean(positionx[positionx != 0]) + 25,
            ]
        )
        ax[1, 0].set_ylim(
            [
                np.mean(positiony[positiony != 0]) - 25,
                np.mean(positiony[positiony != 0]) + 25,
            ]
        )

        ax[1, 1].plot(range(self.data.shape[0]), puncta_num)
        ax[1, 1].set_title("Punctas Number")
        ax[1, 1].set_xlabel("t")
        ax[1, 1].set_ylabel("Punctas")
        ax[2, 0].plot(range(self.data.shape[0]), puncta_size)
        ax[2, 0].set_title("Punctas Size")
        ax[2, 0].set_xlabel("t")
        ax[2, 0].set_ylabel("Size: pixel")

        xx = np.clip(positionx[firstt], 25, self.data.shape[1] - 26)
        yy = np.clip(positiony[firstt], 25, self.data.shape[2] - 26)
        ax[2, 1].imshow(
            self.data[firstt][xx - 25 : xx + 25, yy - 25 : yy + 25]
        )
        ax[2, 1].set_title("Spine in t = %d" % (firstt + 1))

        plt.tight_layout()
        plt.show()
        plt.savefig("spine_%d.png" % spine_number, bbox_inches="tight")


# 这个最终是要弃置的，保留了上述的以前的时序分析，
# 下述一个新类 bugou xin
# 一个新单张数据批量分析的函数


# 由于有filter_small函数,跑的受限，后改。
# from sklearn.decomposition import PCA
import pickle

import cv2
import tifffile as tiff
import utils
from sklearn.neighbors import NearestNeighbors


class SIMDATA:
    def __init__(
        self,
        save_path,  # 本次（批量）处理的根目录+本套数据的标识
        actin_data_path=None,
        spine_mask_path=None,
        dend_mask_path=None,
        presyn_data_path=None,
        presyn_mask_path=None,
        psd_data_path=None,
        psd_mask_path=None,
    ):
        self.actin_data_path = actin_data_path
        self.spine_mask_path = spine_mask_path
        self.dend_mask_path = dend_mask_path
        self.presyn_data_path = presyn_data_path
        self.presyn_mask_path = presyn_mask_path
        self.psd_data_path = psd_data_path
        self.psd_mask_path = psd_mask_path

        self.save_path = str(save_path)
        # 依次尝试读取
        if actin_data_path is not None:
            self.actin_data = tiff.imread(actin_data_path)
            if len(self.actin_data.shape) == 2:
                self.actin_data = self.actin_data[np.newaxis, :, :]
            self.data_shape = self.actin_data.shape
        if spine_mask_path is not None:
            self.spine_mask = tiff.imread(spine_mask_path)
            if len(self.spine_mask.shape) == 2:
                self.spine_mask = self.spine_mask[np.newaxis, :, :]
            if self.actin_data_path is None:
                self.data_shape = self.spine_mask.shape
        if dend_mask_path is not None:
            self.dend_mask = tiff.imread(dend_mask_path)
            if len(self.dend_mask.shape) == 2:
                self.dend_mask = self.dend_mask[np.newaxis, :, :]
            self.have_dend_mask = True
        if presyn_data_path is not None:
            self.presyn_data = tiff.imread(presyn_data_path)
            if len(self.presyn_data.shape) == 2:
                self.presyn_data = self.presyn_data[np.newaxis, :, :]
            self.have_presyn_data = True
        if presyn_mask_path is not None:
            self.presyn_mask = tiff.imread(presyn_mask_path)
            if len(self.presyn_mask.shape) == 2:
                self.presyn_mask = self.presyn_mask[np.newaxis, :, :]
            self.have_presyn_mask = True
        if psd_data_path is not None:
            self.psd_data = tiff.imread(psd_data_path)
            if len(self.psd_data.shape) == 2:
                self.psd_data = self.psd_data[np.newaxis, :, :]
            self.have_psd_data = True
        if psd_mask_path is not None:
            self.psd_mask = tiff.imread(psd_mask_path)
            if len(self.psd_mask.shape) == 2:
                self.psd_mask = self.psd_mask[np.newaxis, :, :]
            self.have_psd_mask = True

    def get_attr_by_name(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None

    def calculate_aspect_ratio(self, binary_mask):
        # binary_mask = (mask_slice == spine_id).astype(np.uint8)
        binary_mask = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        points = contours[0]

        rect = cv2.minAreaRect(points)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        width = rect[1][0]
        height = rect[1][1]
        if width < height:
            length = height
            width = width
        else:
            length = width
            width = height
        aspect_ratio = length / width
        return length, width, aspect_ratio

    def spigle_spine_info_cal(self, spine_id, mask_type="spine"):
        pass

    def single_slice_info_cal(self, slice_id, mask_type="spine", save=False):
        slice_info_save_path = os.path.join(
            self.save_path,
            "slice_infos",
            mask_type,
            f"slice_{slice_id}_info.pkl",
        )
        if not save:
            if os.path.exists(slice_info_save_path):
                with open(slice_info_save_path, "rb") as f:
                    slice_info = pickle.load(f)
                return slice_info
        slice_info = dict()

        mask_slice = self.get_attr_by_name(mask_type + "_mask")[slice_id]
        if self.actin_data_path is not None:
            actin_data_slice = self.actin_data[slice_id]
        if self.psd_mask_path is not None:
            psd_mask_slice = self.psd_mask[slice_id]
            psd_mask_slice = utils.filter_small(psd_mask_slice, threshold=20)

        spine_ids = np.unique(mask_slice)
        spine_ids = spine_ids[spine_ids != 0]

        for spine_id in spine_ids:
            spine_info = dict()

            spine_in_mask = mask_slice == spine_id
            spine_info["size"] = np.count_nonzero(spine_in_mask)
            spine_info["position"] = np.nanmean(
                np.where(spine_in_mask), axis=1
            )

            if self.actin_data_path is not None:
                spine_info["intensity"] = np.nanmean(
                    actin_data_slice[spine_in_mask]
                )
            else:
                spine_info["intensity"] = None

            if self.psd_mask_path is not None:
                _multi = np.multiply(mask_slice == spine_id, psd_mask_slice)
                spine_info["puncta_num"] = np.count_nonzero(np.unique(_multi))
                spine_info["puncta_size"] = np.count_nonzero(_multi)
            else:
                spine_info["puncta_num"] = None
                spine_info["puncta_size"] = None

            length, width, aspect_ratio = self.calculate_aspect_ratio(
                spine_in_mask
            )
            spine_info["length"] = length
            spine_info["width"] = width
            spine_info["aspect_ratio"] = aspect_ratio

            if aspect_ratio > 2.6:
                spine_info["type"] = "filopodia"
            else:
                spine_info["type"] = "spine"

            slice_info[spine_id] = spine_info
        if save:
            if not os.path.exists(
                os.path.join(self.save_path, "slice_infos", mask_type)
            ):
                os.makedirs(
                    os.path.join(self.save_path, "slice_infos", mask_type)
                )
            with open(slice_info_save_path, "wb") as f:
                pickle.dump(slice_info, f)

            # mask_slice1 = np.zeros_like(mask_slice)

            # for spine_id in spine_ids:
            #     mask_slice1[mask_slice == spine_id] = 1 if slice_info[spine_id]['type'] == 'spine' else 2
            # tiff.imwrite(os.path.join(self.rootpath, self.name.split('.')[-2]+ f'_slice_{slice_id}_type.tif'), mask_slice1)
        return slice_info

    def single_slice_analysis(self, slice_id=0, mask_type="spine", save=True):
        slice_info = self.single_slice_info_cal(slice_id, mask_type, save)
        spine_ids = list(slice_info.keys())
        sizes = [slice_info[spine_id]["size"] for spine_id in spine_ids]
        sizes = np.array(sizes)
        positions = [
            slice_info[spine_id]["position"] for spine_id in spine_ids
        ]
        positions = np.array(positions)
        intensities = [
            slice_info[spine_id]["intensity"] for spine_id in spine_ids
        ]
        puncta_nums = [
            slice_info[spine_id]["puncta_num"] for spine_id in spine_ids
        ]
        puncta_sizes = [
            slice_info[spine_id]["puncta_size"] for spine_id in spine_ids
        ]

        types = [slice_info[spine_id]["type"] for spine_id in spine_ids]

        filo_spine_ratio = types.count("filopodia") / len(types)

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(
            positions
        )
        distances, indices = nbrs.kneighbors(positions)
        distances = distances[:, 1]

        return dict(
            sizes=sizes,
            positions=positions,
            intensities=intensities,
            puncta_nums=puncta_nums,
            puncta_sizes=puncta_sizes,
            types=types,
            filo_spine_ratio=filo_spine_ratio,
            distances=distances,
        )


def plot_multi_single_slice_analysis(root, save=True):
    # need actin and data
    actin_end_sig = "_actin_result.tif"
    slice_analysis = []

    for filename in os.listdir(root):
        if filename.endswith(actin_end_sig):
            actinpath = os.path.join(root, filename)
            dataname = filename[: -len(actin_end_sig)] + ".tif"
            datapath = os.path.join(root, dataname)
            SIMDATA1 = SIMDATA(
                datapath=datapath,
                actinpath=actinpath,
            )
            this_slice_analysis = SIMDATA1.single_slice_analysis(save=save)
            this_slice_analysis["name"] = dataname
            this_slice_analysis["data"] = SIMDATA1.data

            slice_analysis.append(this_slice_analysis)

    fig, ax = plt.subplots(
        len(slice_analysis), 5, figsize=(20, 3 * len(slice_analysis))
    )
    for i in range(len(slice_analysis)):
        name = slice_analysis[i]["name"]
        sizes = slice_analysis[i]["sizes"]
        positions = slice_analysis[i]["positions"]
        intensities = slice_analysis[i]["intensities"]
        types = slice_analysis[i]["types"]
        filo_spine_ratio = slice_analysis[i]["filo_spine_ratio"]
        distances = slice_analysis[i]["distances"]

        ax[i, 0].imshow(slice_analysis[i]["data"][0])
        ax[i, 0].set_title(name)
        ax[i, 0].axis("off")

        ax[i, 1].hist(sizes, bins=range(0, 5000, 50))
        mean_size = np.mean(sizes)
        ax[i, 1].set_title("Spine Size")
        ax[i, 1].set_xlabel("Size: pixel")
        ax[i, 1].set_ylabel("Spine Number")
        y_min, y_max = ax[i, 1].get_ylim()
        ax[i, 1].axvline(mean_size, color="r", linestyle="dashed", linewidth=1)
        ax[i, 1].text(
            mean_size, 0.6 * y_max, "Mean Size: %.0f" % mean_size, rotation=0
        )

        ax[i, 2].hist(intensities, bins=range(0, 5000, 50))
        mean_intensity = np.mean(intensities)
        ax[i, 2].set_title("Spine Intensity")
        ax[i, 2].set_xlabel("Intensity: ")
        ax[i, 2].set_ylabel("Spine Number")
        y_min, y_max = ax[i, 2].get_ylim()
        ax[i, 2].axvline(
            mean_intensity, color="r", linestyle="dashed", linewidth=1
        )
        ax[i, 2].text(
            mean_intensity,
            0.6 * y_max,
            "Mean Intensity: %.0f" % mean_intensity,
            rotation=0,
        )

        ax[i, 3].hist(distances.flatten(), bins=range(0, 500, 10))
        mean_distance = np.mean(distances)
        ax[i, 3].set_title("Nearest Distance")
        ax[i, 3].set_xlabel("Distance: pixel")
        ax[i, 3].set_ylabel("Spine Number")
        y_min, y_max = ax[i, 3].get_ylim()
        ax[i, 3].axvline(
            mean_distance, color="r", linestyle="dashed", linewidth=1
        )
        ax[i, 3].text(
            mean_distance,
            0.6 * y_max,
            "Mean Distance: %.0f" % mean_distance,
            rotation=0,
        )

        # ax[i,4].hist(filo_spine_ratio,bins=np.linspace(0
        ax[i, 4].set_title("Filopodia Ratio")
        mean_filo_spine_ratio = np.mean(filo_spine_ratio)
        ax[i, 4].set_xlabel("Ratio: ")
        ax[i, 4].set_ylabel("Spine Number")
        y_min, y_max = ax[i, 4].get_ylim()
        ax[i, 4].axvline(
            mean_filo_spine_ratio, color="r", linestyle="dashed", linewidth=1
        )
        ax[i, 4].text(
            mean_filo_spine_ratio,
            0.6 * y_max,
            "Mean Filopodia Ratio: %.2f" % mean_filo_spine_ratio,
            rotation=0,
        )
    plt.tight_layout()
    fig.savefig(os.path.join(root, "summary.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    #    SIM1 = SIM('/home/lizhuo/disk1/SIM/test1/1.npy','/home/lizhuo/disk1/SIM/test1/1_cp_masks.npy')
    # data:xxx.tif
    # mask:xxx_cp_masks.tif
    data561 = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/2_561.tif"
    mask561 = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/2_561_actin_result_overlap-hwl.tif"
    data640 = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/2_640.tif"
    mask640 = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/2_640_psd_mask-hwl.tif"
    SIM1 = SIM(data561, mask561, mask640)
    # SIM1 = SIM(sys.argv[1],sys.argv[2],sys.argv[3])
    # SIM1.summary_data()
    print("\n")
    SIM1.summary_spine(69)
