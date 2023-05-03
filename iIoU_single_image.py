import torch
import numpy as np
import os
import PIL.Image as Image
import cv2
import os
import argparse

# 26: 7260 [144, 101, 0]
# 24: 2322 [192, 93, 0]
# 27: 26024 [120, 105, 0]
# 28: 75380 [96, 109, 0]
# 25: 1360 [170, 97, 0]
# 33: 6857 [232, 128, 0]
# 32: 4213 [0, 125, 0]
# 31: 105804 [24, 121, 0]

average_sizes = {26: 7260, 24: 2322, 27: 26024, 28: 75380, 25: 1360, 33: 6857, 32: 4213, 31: 105804}
categories = [26, 24, 27, 28, 25, 33, 32, 31]
cate_mapping = {24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
# color mapping only considers G channel in RGB image
color_mapping = {24: 93, 25: 97, 26: 101, 27: 105, 28: 109, 31: 121, 32: 125, 33: 128}
ids2name = {0:'unlabeled',
    1:'ego vehicle',
    2:'rectification border',
    3:'out of roi',
    4:'static',
    5:'dynamic',
    6:'ground',
    7:'road',
    8:'sidewalk',
    9:'parking',
    10:'rail track',
    11:'building',
    12:'wall',
    13:'fence',
    14:'guard rail',
    15:'bridge',
    16:'tunnel',
    17:'pole',
    18:'polegroup',
    19:'traffic light',
    20:'traffic sign',
    21:'vegetation',
    22:'terrain',
    23:'sky',
    24:'person',
    25:'rider',
    26:'car',
    27:'truck',
    28:'bus',
    29:'caravan',
    30:'trailer',
    31:'train',
    32:'motorcycle',
    33:'bicycle',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('gt_dir', help='ground-truth label file directory')
    parser.add_argument('pred_dir', help='predicted label file directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    prediction_dir = args.pred_dir
    gt_dir = args.gt_dir
    txt_out_path = os.path.join(prediction_dir,"iIoU_eval.txt")

    iIoUs = []
    for label in categories:
        TP_pixels = 0
        FN_pixels = 0
        FP_pixels = 0
        average_area = average_sizes[label]
        for prediction_folder in os.listdir(prediction_dir):
            pred_dir_full = os.path.join(prediction_dir, prediction_folder)
            if os.path.isdir(pred_dir_full):
                for prediction_name in os.listdir(pred_dir_full):
                    prediction_path = os.path.join(prediction_dir, prediction_folder, prediction_name)
                    gt_path = os.path.join(gt_dir, prediction_folder, prediction_name.replace("leftImg8bit", "gt_panoptic"))
                    prediction = Image.open(prediction_path)
                    prediction = np.array(prediction)
                    gt = Image.open(gt_path)
                    gt = np.array(gt)
                    pred_mask = prediction == cate_mapping[label]
                    gt_G = gt[:, :, 1]
                    gt_mask = gt_G == color_mapping[label]
                    FP_pixels += np.count_nonzero(pred_mask[~gt_mask]==True)
                    # print(FP_pixels)
                    # unique R channel values of the corresponding masks. e.g. [144, 145] means two instances of corresponding class in this image
                    instance_R = np.unique(gt[gt_mask, :][:, 0])
                    count_instances = len(instance_R)
                    for R_color in instance_R:
                        # print(gt[gt_mask][:, :, 0])
                        instance_map = np.all(gt==(R_color, color_mapping[label], 0), axis=-1)
                        intersection = instance_map[pred_mask] == True
                        intersection_count = np.count_nonzero(intersection)
                        area = np.count_nonzero(instance_map)
                        FN = area - intersection_count
                        weight = average_area/area
                        TP_pixels += weight*intersection_count
                        FN_pixels += weight*FN

        if TP_pixels + FN_pixels + FP_pixels == 0:
            print(ids2name[label], "no instance found")
            continue
        else:
            iIoU = 1.0*TP_pixels/(TP_pixels + FN_pixels + FP_pixels)

        iIoUs.append(iIoU)
        print_temp_str = "The iIoU of " + str(label) + " " + ids2name[label] + " is " + str(iIoU)
        print(print_temp_str)

        with open(txt_out_path, "a") as file_object:
            file_object.write(print_temp_str)

    ave_iIoU = sum(iIoUs) / len(iIoUs)
    print_temp_str = "Average iIoU over all labelled instance classes is " + str(ave_iIoU)
    print(print_temp_str)
    with open(txt_out_path, "a") as file_object:
        file_object.write(print_temp_str)


if __name__ == '__main__':
    main()