import os

import albumentations as A
import numpy as np
import tifffile as tiff
import utils

imagepath = "/share/data/CryoET_Data/lizhuo/SIM-data/actin_train_data/hydata-wlcorrected"
bin_num = 2
target_size = 256


def histogram_equalization(image, max_val):
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, max_val))

    # 计算累积分布函数（CDF）
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())

    # 使用CDF进行均衡化
    equalized_image = np.interp(
        image.flatten(), bins[:-1], cdf_normalized * max_val
    )
    equalized_image = equalized_image.reshape(image.shape)

    return equalized_image.astype(np.uint16)


def random_crop(image, mask, bin, targetsize, imagepath):
    imagesize = image.shape[0]
    resize = imagesize // bin
    num = (resize // targetsize) ** 2 * 16
    # num = 1

    transform = A.Compose(
        [
            A.Resize(resize, resize),
            A.Rotate(limit=180, p=1),
            A.Flip(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(p=0.5, shift_limit=0.02),
                ],
                p=0.2,
            ),
            A.GaussNoise(p=0.2),
            A.RandomCrop(width=targetsize, height=targetsize),
        ]
    )
    # print('image shape:',image.shape)
    # print('mask shape:',mask.shape)

    for j in range(num):
        transformed = transform(image=np.array(image), mask=np.array(mask))
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]

        if np.sum(transformed_mask.astype(np.uint8)) >= 0 or 1:
            tiff.imwrite(
                "/"
                + os.path.join(
                    *imagepath.split("/")[:-1],
                    "imgs",
                    imagepath.split("/")[-1][:-4] + "_" + str(j) + ".tif",
                ),
                transformed_image.astype(np.uint8),
            )
            # Image.fromarray(transformed_image).save('/'+os.path.join(*imagepath.split('/')[:-1],'imgs',imagepath.split('/')[-1][:-4]+'_'+str(j)+'.tif'))
            tiff.imwrite(
                "/"
                + os.path.join(
                    *imagepath.split("/")[:-1],
                    "masks",
                    imagepath.split("/")[-1][:-4] + "_" + str(j) + "_mask.tif",
                ),
                transformed_mask.astype(np.uint8),
            )

    return None


def regular_crop(image, mask, bin, targetsize, imagepath):
    # 不做augmentation。只resize和随机crop
    imagesize = image.shape[0]
    resize = imagesize // bin
    num = (resize // targetsize) ** 2 * 16
    # num = 1

    transform = A.Compose(
        [
            A.Resize(resize, resize),
            A.RandomCrop(width=targetsize, height=targetsize),
        ]
    )

    for j in range(num):
        transformed = transform(image=np.array(image), mask=np.array(mask))
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]

        if np.sum(transformed_mask.astype(np.uint8)) >= 0 or 1:
            tiff.imwrite(
                "/"
                + os.path.join(
                    *imagepath.split("/")[:-1],
                    "imgs",
                    imagepath.split("/")[-1][:-4] + "_" + str(j) + ".tif",
                ),
                transformed_image.astype(np.uint8),
            )
            # Image.fromarray(transformed_image).save('/'+os.path.join(*imagepath.split('/')[:-1],'imgs',imagepath.split('/')[-1][:-4]+'_'+str(j)+'.tif'))
            tiff.imwrite(
                "/"
                + os.path.join(
                    *imagepath.split("/")[:-1],
                    "masks",
                    imagepath.split("/")[-1][:-4] + "_" + str(j) + "_mask.tif",
                ),
                transformed_mask.astype(np.uint8),
            )

    return None


def data_albumentations(root):
    for type in ["train", "test"]:
        if not os.path.exists(root + "/" + type + "_data/imgs"):
            os.makedirs(root + "/" + type + "_data/imgs")
        if not os.path.exists(root + "/" + type + "_data/masks"):
            os.makedirs(root + "/" + type + "_data/masks")
        for file in os.listdir(root + "/" + type + "_data"):
            if (
                file.endswith("-spine.tif")
                or file.endswith("_mask.tif")
                or file.endswith("_actin_result.tif")
            ):
                if file.endswith("_mask.tif"):
                    filename = file[:-9]
                    mask = tiff.imread(
                        root + "/" + type + "_data/" + filename + "_mask.tif"
                    )
                elif file.endswith("_actin_result.tif"):
                    filename = file[:-17]
                    mask = tiff.imread(
                        root
                        + "/"
                        + type
                        + "_data/"
                        + filename
                        + "_actin_result.tif"
                    )
                else:
                    filename = file[:-10]
                    mask = tiff.imread(
                        root + "/" + type + "_data/" + filename + "-spine.tif"
                    )
                image_path = root + "/" + type + "_data/" + filename + ".tif"
                print("processing:", image_path)
                # image = Image.open(root+'/'+type+'_data/'+filename+'.tif')
                image = tiff.imread(image_path)

                image = utils.image_preprocess(
                    image, percentile=99.5, equalize=False
                )
                mask = mask.astype(np.uint8)
                mask[mask > 0] = 1
                if len(mask.shape) == 3:
                    mask = mask[0, :, :]
                if len(image.shape) == 3:
                    image = image[0, :, :]

                if type == "train":
                    random_crop(
                        image,
                        mask,
                        bin=bin_num,
                        targetsize=target_size,
                        imagepath=image_path,
                    )
                else:
                    regular_crop(
                        image,
                        mask,
                        bin=bin_num,
                        targetsize=target_size,
                        imagepath=image_path,
                    )


if __name__ == "__main__":
    data_albumentations(imagepath)
