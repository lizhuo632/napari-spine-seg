# condadl
import copy
import os

# from sklearn.decomposition import PCA
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from natsort import natsorted
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.morphology import skeletonize
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

# import logging
# #import utils

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     filename='log.txt',
#                     filemode='w')


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
        for t in trange(result.shape[0]):
            i = 1
            labeled_image, num_features = measure.label(
                result[t], connectivity=2, return_num=True
            )
            region_props = measure.regionprops(labeled_image)
            for props in region_props:
                if props.area < size_threshold:
                    labeled_image[labeled_image == props.label] = 0
                else:
                    labeled_image[labeled_image == props.label] = i
                    i += 1
            result[t] = labeled_image

    return result


def plot_multi_single_slice_analysis(
    root, load_exist_info=True, save_new=True
):
    # need actin and data
    actin_end_sig = "_actin_result.tif"
    slice_analysis = []

    for filename in natsorted(os.listdir(root)):
        if filename.endswith(actin_end_sig):
            spine_mask_path = os.path.join(root, filename)
            dataname = filename[: -len(actin_end_sig)] + ".tif"
            actin_data_path = os.path.join(root, dataname)
            dend_mask_path = os.path.join(
                root, dataname[:-4] + "_dend_result.tif"
            )
            # dend_mask_path = None
            SIMDATA1 = SIMDATA(
                save_path=root,
                actin_data_path=actin_data_path,
                spine_mask_path=spine_mask_path,
                dend_mask_path=dend_mask_path,
            )
            this_slice_analysis = SIMDATA1.single_slice_analysis(
                load_exist_info=load_exist_info, save_new=save_new
            )
            this_slice_analysis["name"] = dataname
            this_slice_analysis["data"] = SIMDATA1.actin_data

            slice_analysis.append(this_slice_analysis)

    fig, ax = plt.subplots(
        max(2, len(slice_analysis)), 5, figsize=(20, 3 * len(slice_analysis))
    )
    for i in range(len(slice_analysis)):
        name = slice_analysis[i]["name"]
        sizes = slice_analysis[i]["sizes"]
        positions = slice_analysis[i]["positions"]
        intensities = slice_analysis[i]["intensities"]
        types = slice_analysis[i]["types"]
        filo_spine_ratio = slice_analysis[i]["filo_spine_ratio"]
        distances = slice_analysis[i]["distances"]
        aspect_ratios = slice_analysis[i]["aspect_ratios"]
        spine_density = slice_analysis[i]["spine_density"]

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

        ax[i, 3].hist(distances, bins=range(0, 500, 10))
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
            "Mean Distance: %.0f" % mean_distance + " pixel",
            rotation=0,
        )
        if spine_density is not None:
            ax[i, 3].text(
                mean_distance,
                0.4 * y_max,
                "Spine Density: %.2f" % spine_density
                + " Spines / 10um dendrite",
                rotation=0,
            )

        ax[i, 4].hist(aspect_ratios, bins=np.arange(0, 5, 0.1))
        ax[i, 4].set_title("Aspect Ratio")
        mean_filo_spine_ratio = np.mean(filo_spine_ratio)
        ax[i, 4].set_xlabel("Aspect Ratio: ")
        ax[i, 4].set_ylabel("Spine Number")
        y_min, y_max = ax[i, 4].get_ylim()
        ax[i, 4].axvline(2.7, color="g", linestyle="dashed", linewidth=1)
        ax[i, 4].text(
            0,
            0.6 * y_max,
            "Filopodia(Aspect Ratio>2.7) Ratio: %.2f" % mean_filo_spine_ratio,
            rotation=0,
        )
    plt.tight_layout()
    fig.savefig(os.path.join(root, "summary.png"), bbox_inches="tight")
    plt.show()


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
            print("loading actin data:", actin_data_path)

            self.actin_data = tiff.imread(actin_data_path)
            if len(self.actin_data.shape) == 2:
                self.actin_data = self.actin_data[np.newaxis, :, :]
            self.data_shape = self.actin_data.shape

        if spine_mask_path is not None:
            print("loading spine mask:", spine_mask_path)
            self.spine_mask = tiff.imread(spine_mask_path)
            if len(self.spine_mask.shape) == 2:
                self.spine_mask = self.spine_mask[np.newaxis, :, :]
            if self.actin_data_path is None:
                self.data_shape = self.spine_mask.shape
        if dend_mask_path is not None:
            print("loading dend mask:", dend_mask_path)
            self.dend_mask = tiff.imread(dend_mask_path)
            if len(self.dend_mask.shape) == 2:
                self.dend_mask = self.dend_mask[np.newaxis, :, :]
            self.have_dend_mask = True
        if presyn_data_path is not None:
            print("loading presyn data:", presyn_data_path)
            self.presyn_data = tiff.imread(presyn_data_path)
            if len(self.presyn_data.shape) == 2:
                self.presyn_data = self.presyn_data[np.newaxis, :, :]
            self.have_presyn_data = True
        if presyn_mask_path is not None:
            print("loading presyn mask:", presyn_mask_path)
            self.presyn_mask = tiff.imread(presyn_mask_path)
            if len(self.presyn_mask.shape) == 2:
                self.presyn_mask = self.presyn_mask[np.newaxis, :, :]
            self.have_presyn_mask = True
        if psd_data_path is not None:
            print("loading psd data:", psd_data_path)
            self.psd_data = tiff.imread(psd_data_path)
            if len(self.psd_data.shape) == 2:
                self.psd_data = self.psd_data[np.newaxis, :, :]
            self.have_psd_data = True
        if psd_mask_path is not None:
            print("loading psd mask:", psd_mask_path)
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
        if width == 0:
            aspect_ratio = 1
        else:
            aspect_ratio = length / width
        return length, width, aspect_ratio

    def find_dend_touch_point(
        self, position, dend_mask_skeleton_slice, radius=100
    ):
        position = [int(i) for i in position]
        test_search_area = np.zeros([radius * 2, radius * 2])
        test_search_area[radius, radius] = 1
        radius_distance_map = distance_transform_edt(1 - test_search_area)
        dend_nearby_area = np.zeros([radius * 2, radius * 2])
        dend_nearby_area[
            radius
            - min(radius, position[0]) : radius
            + min(radius, dend_mask_skeleton_slice.shape[0] - position[0]),
            radius
            - min(radius, position[1]) : radius
            + min(radius, dend_mask_skeleton_slice.shape[1] - position[1]),
        ] = dend_mask_skeleton_slice[
            max(0, position[0] - radius) : min(
                dend_mask_skeleton_slice.shape[0], position[0] + radius
            ),
            max(0, position[1] - radius) : min(
                dend_mask_skeleton_slice.shape[1], position[1] + radius
            ),
        ]
        multiplied_area = np.multiply(dend_nearby_area, radius_distance_map)
        if np.max(multiplied_area) == 0:
            # nozero_min_value = 0
            nearest_den_point = None
        else:
            nozero_min_value = np.min(multiplied_area[multiplied_area > 0])
            nozero_min_index = np.argwhere(
                multiplied_area == nozero_min_value
            )[0]
            # = np.argmin(np.multiply(dend_nearby_area,radius_distance_map))
            nearest_den_point = [
                nozero_min_index[0] + position[0] - radius,
                nozero_min_index[1] + position[1] - radius,
            ]
        return nearest_den_point

    def single_spine_info_cal(self, spine_id, mask_type="spine"):
        pass

    def single_slice_info_cal(
        self,
        slice_id,
        mask_type="spine",
        needed_props_list=[],
        load_exist_info=True,
        save_new=False,
    ):
        # print('calculating slice:',slice_id)
        slice_info_save_path = os.path.join(
            self.save_path,
            "slice_infos",
            mask_type,
            (self.get_attr_by_name(mask_type + "_mask_path")).split("/")[-1][
                :-4
            ]
            + f"_slice_{slice_id}_info.pkl",
        )
        if load_exist_info and os.path.exists(slice_info_save_path):
            with open(slice_info_save_path, "rb") as f:
                print("loading slice info:", slice_info_save_path)
                slice_info = pickle.load(f)
                # return slice_info

        else:
            print("calculating slice:", slice_id)
            print("needed_props_list:", needed_props_list)
            slice_info = dict()

            mask_slice = self.get_attr_by_name(mask_type + "_mask")[slice_id]
            if self.actin_data_path is not None:
                actin_data_slice = self.actin_data[slice_id]
            if self.psd_mask_path is not None:
                psd_mask_slice = self.psd_mask[slice_id]
                psd_mask_slice = filter_small(psd_mask_slice, threshold=20)
            if self.presyn_mask_path is not None:
                presyn_mask_slice = self.presyn_mask[slice_id]
                # presyn_mask_slice = filter_small(presyn_mask_slice,threshold=20)
            if self.dend_mask_path is not None:
                dend_mask_slice = self.dend_mask[slice_id]
                # dend_mask_slice = filter_small(dend_mask_slice,threshold=80)
                dend_mask_binary_slice = dend_mask_slice > 0
                dend_mask_skeleton_slice = skeletonize(dend_mask_binary_slice)
                # dend_distance_map = distance_transform_edt(dend_mask_binary)
                # dend_skeleton_with_radius = np.multiply(dend_mask_skeleton_slice,dend_distance_map)

                dend_length = np.sum(dend_mask_skeleton_slice)

            spine_ids = np.unique(mask_slice)
            spine_ids = spine_ids[spine_ids != 0]

            for spine_id in spine_ids:
                spine_info = dict()

                # mask
                spine_in_mask = mask_slice == spine_id
                spine_info["size"] = np.count_nonzero(spine_in_mask)
                spine_info["position"] = np.nanmean(
                    np.where(spine_in_mask), axis=1
                )

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

                # actin_data
                if self.actin_data_path is not None:
                    spine_info["intensity"] = np.nanmean(
                        actin_data_slice[spine_in_mask]
                    )
                else:
                    spine_info["intensity"] = None

                # psd_mask
                if self.psd_mask_path is not None:
                    _multi = np.multiply(
                        mask_slice == spine_id, psd_mask_slice
                    )
                    spine_info["puncta_num"] = np.count_nonzero(
                        np.unique(_multi)
                    )
                    spine_info["puncta_size"] = np.count_nonzero(_multi)
                else:
                    spine_info["puncta_num"] = None
                    spine_info["puncta_size"] = None

                # dend_mask
                if self.dend_mask_path is not None:
                    spine_info["dend_overlap"] = np.count_nonzero(
                        np.multiply(mask_slice == spine_id, dend_mask_slice)
                    )
                    spine_info["dend_length"] = dend_length
                else:
                    spine_info["dend_overlap"] = None
                    spine_info["dend_length"] = None

                # presyn_mask
                if self.presyn_mask_path is not None:
                    spine_info["presyn_size"] = np.count_nonzero(
                        np.multiply(mask_slice == spine_id, presyn_mask_slice)
                    )
                else:
                    spine_info["presyn_size"] = None

                slice_info[spine_id] = spine_info

        # 如果slice_info没有‘distance'key就计算
        # distance需要所有点的信息所以额外放这里
        if "distance" not in next(iter(slice_info.values())).keys():
            if "distance" in needed_props_list or "all" in needed_props_list:
                mask_slice = self.get_attr_by_name(mask_type + "_mask")[
                    slice_id
                ]
                spine_ids = np.unique(mask_slice)
                spine_ids = spine_ids[spine_ids != 0]
                positions = [
                    slice_info[spine_id]["position"] for spine_id in spine_ids
                ]
                positions = np.array(positions)
                nbrs = NearestNeighbors(
                    n_neighbors=2, algorithm="ball_tree"
                ).fit(positions)
                distances, indices = nbrs.kneighbors(positions)
                distances = distances[:, 1]
                for i, spine_id in enumerate(spine_ids):
                    slice_info[spine_id]["distance"] = distances[i]
            else:
                for spine_id in spine_ids:
                    slice_info[spine_id]["distance"] = None

        if (
            "dend_touch_point" not in next(iter(slice_info.values())).keys()
            and self.dend_mask_path is not None
        ):
            if (
                "dend_touch_point" in needed_props_list
                or "all" in needed_props_list
            ):
                mask_slice = self.get_attr_by_name(mask_type + "_mask")[
                    slice_id
                ]
                spine_ids = np.unique(mask_slice)
                spine_ids = spine_ids[spine_ids != 0]
                positions = [
                    slice_info[spine_id]["position"] for spine_id in spine_ids
                ]
                positions = np.array(positions)

                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     futures = []
                for i, spine_id in enumerate(spine_ids):
                    position = slice_info[spine_id]["position"]
                    nearest_den_point = self.find_dend_touch_point(
                        position, dend_mask_skeleton_slice, radius=100
                    )
                    slice_info[spine_id][
                        "dend_touch_point"
                    ] = nearest_den_point
            else:
                for spine_id in spine_ids:
                    slice_info[spine_id]["dend_touch_point"] = None

        if save_new:
            if not os.path.exists(os.path.dirname(slice_info_save_path)):
                os.makedirs(os.path.dirname(slice_info_save_path))
            with open(slice_info_save_path, "wb") as f:
                pickle.dump(slice_info, f)

            # mask_slice1 = np.zeros_like(mask_slice)

            # for spine_id in spine_ids:
            #     mask_slice1[mask_slice == spine_id] = 1 if slice_info[spine_id]['type'] == 'spine' else 2
            # tiff.imwrite(os.path.join(self.rootpath, self.name.split('.')[-2]+ f'_slice_{slice_id}_type.tif'), mask_slice1)
        return slice_info

    def single_slice_analysis(
        self,
        slice_id=0,
        mask_type="spine",
        load_exist_info=True,
        save_new=False,
    ):
        # 待更新
        slice_info = self.single_slice_info_cal(
            slice_id, mask_type, load_exist_info, save_new
        )
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

        aspect_ratios = [
            slice_info[spine_id]["aspect_ratio"] for spine_id in spine_ids
        ]
        types = [slice_info[spine_id]["type"] for spine_id in spine_ids]
        filo_spine_ratio = types.count("filopodia") / len(types)

        # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(positions)
        # distances, indices = nbrs.kneighbors(positions)
        # distances = distances[:,1]
        distances = [
            slice_info[spine_id]["distance"] for spine_id in spine_ids
        ]

        dend_touch_points = [
            slice_info[spine_id]["dend_touch_point"] for spine_id in spine_ids
        ]

        spine_num = sum(
            1 for point in dend_touch_points if point is not None
        )  # len(spine_ids)
        dend_length_pixel = slice_info[spine_ids[0]]["dend_length"]
        if dend_length_pixel is not None:
            # print(dend_length_pixel)
            pixel_size = 32.6797  # nm 这个好像其实不应该写在这里

            spine_density = spine_num / (
                dend_length_pixel * pixel_size / 10000
            )  # spines / 10um
        else:
            spine_density = None

        return dict(
            sizes=sizes,
            positions=positions,
            intensities=intensities,
            puncta_nums=puncta_nums,
            puncta_sizes=puncta_sizes,
            aspect_ratios=aspect_ratios,
            types=types,
            filo_spine_ratio=filo_spine_ratio,
            distances=distances,
            spine_density=spine_density,
        )

    def slices_infos_cal(
        self,
        mask_type="spine",
        needed_props_list=[],
        load_exist_info=True,
        save_new=True,
    ):
        # print('calculating slices infos')
        slices_infos_save_path = os.path.join(
            self.save_path, "slices_infos", mask_type + "_slices_infos.pkl"
        )
        if load_exist_info and os.path.exists(slices_infos_save_path):
            with open(slices_infos_save_path, "rb") as f:
                print("loading slices infos:", slices_infos_save_path)
                slices_infos = pickle.load(f)

        else:
            print("calculating slices infos")
            slices_infos = []
            for i in trange(self.data_shape[0]):
                slice_info = self.single_slice_info_cal(
                    i, mask_type, needed_props_list, load_exist_info, save_new
                )
                slices_infos.append(slice_info)

        if save_new:
            # 如果slice_infos_save_path的目录不存在，就创建

            if not os.path.exists(os.path.dirname(slices_infos_save_path)):
                os.makedirs(os.path.dirname(slices_infos_save_path))
            with open(slices_infos_save_path, "wb") as f:
                pickle.dump(slices_infos, f)
        return slices_infos  # list of dict

    def spines_infos_cal(
        self,
        mask_type="spine",
        needed_props_list=[],
        load_exist_info=True,
        save_new=True,
    ):

        spines_infos_save_path = os.path.join(
            self.save_path, "spines_infos", mask_type + "_spines_infos.pkl"
        )

        if load_exist_info and os.path.exists(spines_infos_save_path):
            with open(spines_infos_save_path, "rb") as f:
                print("loading spines infos:", spines_infos_save_path)
                spines_infos_list = pickle.load(f)
        else:
            # spine_info_keys = list(slices_infos[0][0].keys()) ????
            slices_infos = self.slices_infos_cal(
                mask_type,
                needed_props_list=needed_props_list,
                load_exist_info=load_exist_info,
                save_new=save_new,
            )
            print(len(slices_infos))
            print("calculating spines infos")
            spines_infos_list = []
            # spine_infos[0]的spin_id是1
            spine_ids = np.unique(self.get_attr_by_name(mask_type + "_mask"))
            # spine_num = len(np.unique(self.get_attr_by_name(mask_type+'_mask')))
            #!!!待修改。reid？
            spine_info_keys = next(iter(slices_infos[0].values())).keys()
            empty_spine_info_dict = dict()
            empty_spine_info_dict["ID"] = None
            empty_spine_info_dict["t"] = []
            for key in spine_info_keys:
                empty_spine_info_dict[key] = []
            ID2spines_infos_list_index_dict = dict()
            for i, spine_id in enumerate(spine_ids):
                ID2spines_infos_list_index_dict[spine_id] = i - 1
            # spine_num = max(np.unique(self.get_attr_by_name(mask_type+'_mask')))
            for spine_id in spine_ids:
                spine_info_dict = copy.deepcopy(empty_spine_info_dict)
                # spine_info_dict = empty_spine_info_dict.copy.deepcopy()
                spine_info_dict["ID"] = spine_id
                spines_infos_list.append(spine_info_dict)

            # print('spine_1',spines_infos_list[1])
            for t in trange(
                self.get_attr_by_name(mask_type + "_mask").shape[0]
            ):
                # print('t:',t)
                slice_info_dict = slices_infos[t]
                spine_id_in_slice = list(slice_info_dict.keys())
                for spine_id in spine_id_in_slice:
                    # if spine_id != 2:
                    #     continue
                    this_slice_spine_info = slice_info_dict[spine_id]
                    spines_infos_list[
                        ID2spines_infos_list_index_dict[spine_id]
                    ]["t"].append(t)
                    # print('spine_id:',spine_id)
                    # print('t',t)
                    for key in this_slice_spine_info.keys():
                        spines_infos_list[
                            ID2spines_infos_list_index_dict[spine_id]
                        ][key].append(this_slice_spine_info[key])
                #         print(key,':',this_slice_spine_info[key])
                # if t == 2:
                #     break
            # print('spine_1',spines_infos_list[1])
            # print("enmty_spine_info_dict",empty_spine_info_dict)

            # spine_ids = list(slice_info.keys())
            # for spine_id in spine_ids:
            #     print(spine_id)
            #     spine_info = spines_infos_list[spine_id-1]
            #     if spine_id in slice_info:
            #         spine_info['t'].append(t)
            #         spine_info['size'].append(slice_info[spine_id]['size'])
            #         spine_info['position'].append(slice_info[spine_id]['position'])
            #         spine_info['intensity'].append(slice_info[spine_id]['intensity'])
            #         spine_info['puncta_num'].append(slice_info[spine_id]['puncta_num'])
            #         spine_info['puncta_size'].append(slice_info[spine_id]['puncta_size'])
            #         spine_info['dend_overlap'].append(slice_info[spine_id]['dend_overlap'])
            #         spine_info['presyn_size'].append(slice_info[spine_id]['presyn_size'])
            #         spine_info['length'].append(slice_info[spine_id]['length'])
            #         spine_info['width'].append(slice_info[spine_id]['width'])
            #         spine_info['aspect_ratio'].append(slice_info[spine_id]['aspect_ratio'])
            #         spine_info['type'].append(slice_info[spine_id]['type'])
            #         spine_info['distance'].append(slice_info[spine_id]['distance'])
            #         spines_infos_list[spine_id-1] = spine_info

        if save_new:
            if not os.path.exists(os.path.dirname(spines_infos_save_path)):
                os.makedirs(os.path.dirname(spines_infos_save_path))
            with open(spines_infos_save_path, "wb") as f:
                pickle.dump(spines_infos_list, f)
        return spines_infos_list, ID2spines_infos_list_index_dict, slices_infos

    def time_series_info_cal(
        self, mask_type="spine", load_exist_info=True, save_new=False
    ):
        time_series_info_save_path = os.path.join(
            self.save_path,
            "time_series_infos",
            mask_type + "_time_series_infos.pkl",
        )
        if load_exist_info and os.path.exists(time_series_info_save_path):
            with open(time_series_info_save_path, "rb") as f:
                print("loading time series info:", time_series_info_save_path)
                time_series_info = pickle.load(f)

        else:
            print("calculating time series info")
            slices_infos = self.slices_infos_cal(
                mask_type, load_exist_info, save_new
            )
            avg_sizes = []
            avg_intensities = []
            avg_puncta_nums = []
            avg_puncta_sizes = []
            avg_dend_overlaps = []
            avg_presyn_sizes = []
            avg_filo_spine_ratios = []
            avg_distances = []
            for slice_info in slices_infos:
                spine_ids = list(slice_info.keys())
                sizes = [
                    slice_info[spine_id]["size"] for spine_id in spine_ids
                ]
                avg_sizes.append(np.mean(sizes))
                intensities = [
                    slice_info[spine_id]["intensity"] for spine_id in spine_ids
                ]
                avg_intensities.append(np.mean(intensities))
                puncta_nums = [
                    slice_info[spine_id]["puncta_num"]
                    for spine_id in spine_ids
                ]
                avg_puncta_nums.append(np.mean(puncta_nums))
                puncta_sizes = [
                    slice_info[spine_id]["puncta_size"]
                    for spine_id in spine_ids
                ]
                avg_puncta_sizes.append(np.mean(puncta_sizes))
                dend_overlaps = [
                    slice_info[spine_id]["dend_overlap"]
                    for spine_id in spine_ids
                ]
                avg_dend_overlaps.append(np.mean(dend_overlaps))
                presyn_sizes = [
                    slice_info[spine_id]["presyn_size"]
                    for spine_id in spine_ids
                ]
                avg_presyn_sizes.append(np.mean(presyn_sizes))
                filo_spine_ratios = [
                    slice_info[spine_id]["type"] for spine_id in spine_ids
                ]
                avg_filo_spine_ratios.append(
                    filo_spine_ratios.count("filopodia")
                    / len(filo_spine_ratios)
                )
                distances = [
                    slice_info[spine_id]["distance"] for spine_id in spine_ids
                ]
                avg_distances.append(np.mean(distances))
            time_series_info = dict(
                avg_sizes=avg_sizes,
                avg_intensities=avg_intensities,
                avg_puncta_nums=avg_puncta_nums,
                avg_puncta_sizes=avg_puncta_sizes,
                avg_dend_overlaps=avg_dend_overlaps,
                avg_presyn_sizes=avg_presyn_sizes,
                avg_filo_spine_ratios=avg_filo_spine_ratios,
                avg_distances=avg_distances,
            )

        if save_new:
            if not os.path.exists(os.path.dirname(time_series_info_save_path)):
                os.makedirs(os.path.dirname(time_series_info_save_path))
            with open(time_series_info_save_path, "wb") as f:
                pickle.dump(time_series_info, f)
            fig, ax = plt.subplots(7, 1, figsize=(10, 20))
            ax[0].plot(avg_sizes)
            ax[0].set_title("Spine Size")
            ax[1].plot(avg_intensities)
            ax[1].set_title("Spine Intensity")
            ax[2].plot(avg_puncta_nums)
            ax[2].set_title("Puncta Number")
            ax[3].plot(avg_puncta_sizes)
            ax[3].set_title("Puncta Size")
            ax[4].plot(avg_dend_overlaps)
            ax[4].set_title("Dendritic Overlap")
            ax[5].plot(avg_presyn_sizes)
            ax[5].set_title("Presynaptic Size")
            ax[6].plot(avg_filo_spine_ratios)
            ax[6].set_title("Filopodia Ratio")
            plt.tight_layout()
            fig.savefig(
                os.path.join(
                    self.save_path,
                    self.save_path.split("/")[-1] + "-time_series_summary.png",
                ),
                bbox_inches="tight",
            )


# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, default='test')
# args = parser.parse_args()
# name = args.name
# save_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/time-series_sbatch',name)
# actin_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin',name+'-actin.tif')
# spine_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-train12-weights2-unet_180-th_05-ft_80-spine',name+'-actin_spine_result_track.tif')
# dend_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-traindata-weights3-unet_150-th_01-ft_500-dend',name+'-actin_dend_result.tif')
# presyn_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn',name+'-presyn.tif')
# presyn_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn/0result-train12-weights2-unet_180-th_05-ft_80-presyn',name+'-presyn_presyn_result.tif')
# psd_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd.tif')
# psd_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd_psd_result.tif')
# #为什么光是读完就要 36min？/5min
# SIMDATA1 = SIMDATA(save_path=save_path,actin_data_path=actin_data_path,spine_mask_path=spine_mask_path,dend_mask_path=dend_mask_path,presyn_data_path=presyn_data_path,presyn_mask_path=presyn_mask_path,psd_data_path=psd_data_path,psd_mask_path=psd_mask_path)
# #SIMDATA1.spines_infos_cal(mask_type='spine',load_exist_info = True ,save_new=True)
# SIMDATA1.time_series_info_cal(mask_type='spine',load_exist_info = True ,save_new=True)

# for filename in os.listdir('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin'):
#     if filename.endswith('-actin.tif'):
#         name = filename.split('-')[0]
#         save_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/time-series_sbatch',name)
#         actin_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin',name+'-actin.tif')
#         spine_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-train12-weights2-unet_180-th_05-ft_80-spine',name+'-actin_spine_result_track.tif')
#         dend_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-traindata-weights3-unet_150-th_01-ft_500-dend',name+'-actin_dend_result.tif')
#         presyn_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn',name+'-presyn.tif')
#         presyn_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn/0result-train12-weights2-unet_180-th_05-ft_80-presyn',name+'-presyn_presyn_result.tif')
#         psd_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd.tif')
#         psd_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd_psd_result.tif')
#         #为什么光是读完就要 36min？/5min
#         SIMDATA1 = SIMDATA(save_path=save_path,actin_data_path=actin_data_path,spine_mask_path=spine_mask_path,dend_mask_path=dend_mask_path,presyn_data_path=presyn_data_path,presyn_mask_path=presyn_mask_path,psd_data_path=psd_data_path,psd_mask_path=psd_mask_path)
#         SIMDATA1.spines_infos_cal(mask_type='spine',save=True)


# plot_multi_single_slice_analysis('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/test/dend_length_test')
# plot_multi_single_slice_analysis('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/single-slice-comparision/hwl',False,True)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, default='test')
    # args = parser.parse_args()
    # name = args.name
    # save_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/time-series_sbatch',name)
    # actin_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin',name+'-actin.tif')
    # spine_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-train12-weights2-unet_180-th_05-ft_80-spine',name+'-actin_spine_result_track.tif')
    # dend_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-traindata-weights3-unet_150-th_01-ft_500-dend',name+'-actin_dend_result.tif')
    # presyn_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn',name+'-presyn.tif')
    # presyn_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn/0result-train12-weights2-unet_180-th_05-ft_80-presyn',name+'-presyn_presyn_result.tif')
    # psd_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd.tif')
    # psd_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd_psd_result.tif')
    # #为什么光是读完就要 36min？/5min
    # SIMDATA1 = SIMDATA(save_path=save_path,actin_data_path=actin_data_path,spine_mask_path=spine_mask_path,dend_mask_path=dend_mask_path,presyn_data_path=presyn_data_path,presyn_mask_path=presyn_mask_path,psd_data_path=psd_data_path,psd_mask_path=psd_mask_path)
    # #SIMDATA1.spines_infos_cal(mask_type='spine',load_exist_info = True ,save_new=True)
    # SIMDATA1.time_series_info_cal(mask_type='spine',load_exist_info = True ,save_new=True)

    # for filename in os.listdir('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin'):
    #     if filename.endswith('-actin.tif'):
    #         name = filename.split('-')[0]
    #         save_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/time-series_sbatch',name)
    #         actin_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin',name+'-actin.tif')
    #         spine_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-train12-weights2-unet_180-th_05-ft_80-spine',name+'-actin_spine_result_track.tif')
    #         dend_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-traindata-weights3-unet_150-th_01-ft_500-dend',name+'-actin_dend_result.tif')
    #         presyn_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn',name+'-presyn.tif')
    #         presyn_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/presyn/0result-train12-weights2-unet_180-th_05-ft_80-presyn',name+'-presyn_presyn_result.tif')
    #         psd_data_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd.tif')
    #         psd_mask_path = os.path.join('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/psd',name+'-psd_psd_result.tif')
    #         #为什么光是读完就要 36min？/5min
    #         SIMDATA1 = SIMDATA(save_path=save_path,actin_data_path=actin_data_path,spine_mask_path=spine_mask_path,dend_mask_path=dend_mask_path,presyn_data_path=presyn_data_path,presyn_mask_path=presyn_mask_path,psd_data_path=psd_data_path,psd_mask_path=psd_mask_path)
    #         SIMDATA1.spines_infos_cal(mask_type='spine',save=True)

    # plot_multi_single_slice_analysis('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/test/dend_length_test')
    # plot_multi_single_slice_analysis('/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/analysis/single-slice-comparision/hwl',False,True)
    actin_data__path = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/2_561.tif"
    spine_mask_path = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/2_561_actin_result_overlap-hwl.tif"
    psd_mask__path = "/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/2_640_psd_mask.tif"

    SIM2 = SIMDATA(
        save_path="/share/data/CryoET_Data/lizhuo/work_with_wenlan/manual_correction/2/analysissbtach",
        actin_data_path=actin_data__path,
        spine_mask_path=spine_mask_path,
        psd_mask_path=psd_mask__path,
    )
    needed_props_list = [
        "size",
        "position",
        "intensity",
        "puncta_num",
        "puncta_size",
    ]
    spine_infos = SIM2.spines_infos_cal(
        mask_type="spine",
        needed_props_list=needed_props_list,
        load_exist_info=False,
        save_new=True,
    )
