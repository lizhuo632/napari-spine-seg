# 由于lib，只能在本地运行（base/condanpr）

import os

import btrack
import numpy as np

# import napari
import tifffile as tiff
import utils
from skimage.io import imread

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def find_nearest_nonzero_value(data, point):
    if data[point[0], point[1]] != 0:
        return data[point[0], point[1]]
    # Get the coordinates of non-zero points
    nonzero_coords = np.transpose(np.nonzero(data))

    # Calculate the Euclidean distance between the given point and all non-zero points
    distances = np.linalg.norm(nonzero_coords - point, axis=1)

    # Find the index of the nearest non-zero point
    nearest_index = np.argmin(distances)

    # Get the value of the nearest non-zero point
    nearest_value = data[
        nonzero_coords[nearest_index][0], nonzero_coords[nearest_index][1]
    ]

    return nearest_value


def track(mask_path, save=True):

    tracks_save_path = mask_path[:-4] + "_tracks.h5"
    segmentation = imread(mask_path)

    if os.path.exists(tracks_save_path) and not save:
        with btrack.io.HDF5FileHandler(
            tracks_save_path, "r", obj_type="obj_type_1"
        ) as reader:
            tracks = reader.tracks

    else:
        # 将 segmentation 转换为对象
        objects = btrack.utils.segmentation_to_objects(
            segmentation,
            properties=("area", "major_axis_length", "orientation"),
        )

        with btrack.BayesianTracker() as tracker:
            # 配置追踪器
            tracker.configure(
                "/share/data/CryoET_Data/lizhuo/Code/dl/cell_config.json"
            )

            tracker.update_method = (
                btrack.constants.BayesianUpdates.APPROXIMATE
            )  # 3min,和exact一致，怀疑没有用到
            # tracker.max_search_radius = 80
            # 将 objects 添加到追踪器
            tracker.append(objects)

            # 设置追踪范围
            tracker.volume = (
                (0, segmentation.shape[2]),
                (0, segmentation.shape[1]),
            )

            # 进行交互式追踪
            tracker.track(
                tracking_updates=["motion", "visual"]
            )  # _interactive(step_size=100)

            # 生成假设并优化追踪结果
            tracker.optimize()
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            # 将结果导出为 HDF5 文件
            tracker.export(tracks_save_path, obj_type="obj_type_1")

            # 获取追踪数据
            tracks = tracker.tracks

            data, properties, graph = tracker.to_napari()

    # viewer = napari.Viewer()
    # viewer.add_labels(segmentation)
    # viewer.add_tracks(data, properties=properties, graph=graph)

    id_idx = dict()
    for i in range(len(tracks)):
        id_idx[tracks[i].ID] = i
    new_seg = segmentation.copy().astype(np.uint16)
    new_value = len(tracks)
    exist_ts = []
    for track in list(reversed(tracks)):
        len_track = len(track)
        exist_ts.append(len_track)
        _pre = 0
        print(track.ID)

        if len(track.children) > 0 or track.generation >= 2:
            rootid = track.root
        elif len_track <= 3:
            if track.generation >= 1:
                rootid = track.root
            else:
                rootid = 0
                # if track.t[i] > 0 :
                #     #_pre = 1 #会导致一张里有多个同id物体
                #     pass
        elif max(track.t) >= 77:
            rootid = track.root
        else:
            rootid = track.ID

        # 上述在建立track之间的联系。你不能通过btrack本身进行？
        # 下述在给同track涂色

        for i in range(len_track):
            point = (int(track.y[i]) - 1, int(track.x[i]) - 1)
            value = find_nearest_nonzero_value(
                segmentation[int(track.t[i]) - _pre], point
            )
            # print('value:',value)

            if value != 0:
                # count2 += 1
                # if track.ID == 9:
                new_seg[track.t[i]][
                    segmentation[track.t[i]] == value
                ] = rootid  # new_value
            else:
                print("error")

    re_id_new_seg = utils.reset_id(new_seg)
    tiff.imwrite(mask_path.split(".")[0] + "_track.tif", re_id_new_seg)


if __name__ == "__main__":
    # mask_path = '/share/data/CryoET_Data/lizhuo/trace/btracktest/l6-actin_actin_result_1024.tif'

    # track(mask_path,save=True)

    root_path = "/share/data/CryoET_Data/lizhuo/SIM-process/240924/data/actin/0result-train12-weights2-unet_180-th_05-ft_80"

    for mask in os.listdir(root_path):
        if mask.endswith(".tif"):
            # if mask == 'l4-actin_actin_result.tif' or mask == 'l6-actin_actin_result.tif':
            #     continue
            mask_path = os.path.join(root_path, mask)
            track(mask_path, save=True)
