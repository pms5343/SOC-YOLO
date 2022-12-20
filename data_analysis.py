# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
import itertools
import os


import numpy as np
import yaml
from dotmap import DotMap
from tabulate import tabulate


def load_dota_instances(image_data_path: str, label_data_path: str, class_names: list):
    """
    Load  DOTA detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    fileids = os.listdir(label_data_path)

    dicts = []
    for fileid in fileids:
        fileid = fileid.split(".")[0]
        anno_file = os.path.join(label_data_path, fileid + ".txt")
        jpeg_file = os.path.join(image_data_path, fileid + ".jpg")
        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
        }
        instances = []
        anno_file = open(anno_file, "r", encoding="utf-8", errors="ignore")


        mystr = anno_file.readline()

        label_file_list = mystr.split(' ')

        bbox = []
        bbox.append(float(label_file_list[1]))
        bbox.append(float(label_file_list[2]))
        bbox.append(float(label_file_list[3]))
        bbox.append(float(label_file_list[4]))
        b1_x1, b1_x2 = bbox[0] - bbox[2] / 2, bbox[0] + bbox[2] / 2
        b1_y1, b1_y2 = bbox[1] - bbox[3] / 2, bbox[1] + bbox[3] / 2
        bbox = [b1_x1,b1_y1,b1_x2,b1_y2]
        
        instances.append(
            {"category_id": '0', "bbox": bbox, "file_name2": fileid}
        )
        r["annotations"] = instances
        dicts.append(r)

    return dicts


def _print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(table)

def _print_instances_small_middle_larg_histogram(dataset_dicts, class_names):
    """
     Args:
         dataset_dicts (list[dict]): list of dataset dicts.
         class_names (list[str]): list of class names (zero-indexed).
     """
    small = []   
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram_total = np.zeros((num_classes,), dtype=np.int)
    histogram_tiny = np.zeros((num_classes,), dtype=np.int)
    histogram_small = np.zeros((num_classes,), dtype=np.int)
    histogram_medium = np.zeros((num_classes,), dtype=np.int)
    histogram_large = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]

        classes_total = [x["category_id"] for x in annos]

        classes_tiny2 = [x["file_name2"] for x in annos if
                         (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= 0.001]
        while '' in classes_tiny2:
            classes_tiny2.remove('')

        str = ''
        separtor = ','
        for idx, val in enumerate(classes_tiny2):
            str += val + ('' if idx == len(classes_tiny2) -1 else separtor)
            os.renames("/home/pms5343/python/yolor-main/data/small_3/images/" +str+".jpg", "/home/pms5343/python/yolor-main/data/tiny_3/" + "images/" +str+".jpg")



        classes_small2 = [x["file_name2"] for x in annos if
                         0.001 <= (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= 0.003]



        classes_medium2 = [x["file_name2"] for x in annos if
                          0.003 <= (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= 0.03]


        classes_large2 = [x["file_name2"] for x in annos if
                         (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) >= 0.03]


#            text_file = open(r'/home/pms5343/python/large.csv', 'a')
#            for i in histogram_total:
#                data = str + "\n" % i
#                text_file.write(data)
        classes_tiny = [x["category_id"] for x in annos if
                        (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= 0.001]   
        classes_small = [x["category_id"] for x in annos if
                         0.001 <= (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= 0.003]
        classes_medium = [x["category_id"] for x in annos if
                          0.003 <= (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= 0.03]
        classes_large = [x["category_id"] for x in annos if
                         (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) >= 0.03]
        
        histogram_total += np.histogram(classes_total, bins=hist_bins)[0]
        histogram_tiny += np.histogram(classes_tiny, bins=hist_bins)[0]
        histogram_small += np.histogram(classes_small, bins=hist_bins)[0]
        histogram_medium += np.histogram(classes_medium, bins=hist_bins)[0]
        histogram_large += np.histogram(classes_large, bins=hist_bins)[0]

    # histogram=histogram+histogram1
    histogram = np.concatenate((histogram_total, histogram_tiny, histogram_small, histogram_medium, histogram_large)).reshape(5,
                                                                                                              num_classes).T

    # 原始数组转置a.T

    N_COLS = min(5, len(class_names) * 5)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v[0]), int(v[1]), int(v[2]), int(v[3])] for i, v in
                          enumerate(histogram)])
    )
    # total_num_instances = sum(data[1::2])
    total_num_instances = 0
    total_small_instances = 0
    total_medium_instances = 0
    total_large_instances = 0
    for i in range(0, num_classes):
        total_num_instances = total_num_instances + data[i * 5 + 1]
        total_small_instances = total_small_instances + data[i * 5 + 2]
        total_medium_instances = total_medium_instances + data[i * 5 + 3]
        total_large_instances = total_large_instances + data[i * 5 + 4]
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    # if num_classes > 1:
    data.extend(["total", total_num_instances, total_small_instances, total_medium_instances, total_large_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instance", "small", "medium", "large"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(table)


def get_classname(train_data_path):
    """
   Args:
       train_data_path (str)]): train data path
   """
    train_data_yml_name = os.path.basename(train_data_path)
    #with open(os.path.join(train_data_path, train_data_yml_name + '.sda')) as f:
    with open(os.path.join('/home/pms5343/python/yolor-main/data/예비.yaml')) as f:        
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        voc_config = DotMap(config_dict)
        classes = voc_config.dataset.get('classes')
        # del (classes[0])
    return classes


def conunt_instances_class_number(dataset_dicts, class_names):
    """
   Args:
       dataset_dicts (list[dict]): list of dataset dicts.
       class_names (list[str]): list of class names (zero-indexed).
   """
    for dataset_name, dicts in zip(class_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    has_instances = "annotations" in dataset_dicts[0]
    if has_instances:
        try:
            _print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass


def count_instances_class_small_middle_large_number(dataset_dicts, class_names):
    """
   Args:
       dataset_dicts (list[dict]): list of dataset dicts.
       class_names (list[str]): list of class names (zero-indexed).
   """
    for dataset_name, dicts in zip(class_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    has_instances = "annotations" in dataset_dicts[0]
    if has_instances:
        try:
            _print_instances_small_middle_larg_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass


def _print_instances_class_aspect_ratio_histogram(dataset_dicts, class_names):
    """
    总的类别长宽比
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    classes_total_h_w = {}
    classes_total_num = {}
    classes_total_total = {}
    classes_total_max = {}
    # 通过类别初始化字典
    for i in range(0, num_classes):
        classes_total_h_w[i] = 0
        classes_total_num[i] = 0
        classes_total_total[i] = 0
        classes_total_max[i] = 0
    for entry in dataset_dicts:
        annos = entry["annotations"]
        for x in annos:
            if ((x['bbox'][3] - x['bbox'][1]) >= (x['bbox'][2] - x['bbox'][0])) & ((x['bbox'][2] - x['bbox'][0]) > 0):
                ratio_ = round(x['bbox'][3] - x['bbox'][1]) / (x['bbox'][2] - x['bbox'][0])
                classes_total_h_w[x["category_id"]] = classes_total_h_w[x["category_id"]] + ratio_
                classes_total_max[x["category_id"]] = max(classes_total_max[x["category_id"]], ratio_)
            elif ((x['bbox'][2] - x['bbox'][0]) >= (x['bbox'][3] - x['bbox'][1])) & ((x['bbox'][3] - x['bbox'][1]) > 0):
                ratio_ = round(x['bbox'][2] - x['bbox'][0]) / (x['bbox'][3] - x['bbox'][1])
                classes_total_h_w[x["category_id"]] = classes_total_h_w[x["category_id"]] + ratio_
                classes_total_max[x["category_id"]] = max(classes_total_max[x["category_id"]], ratio_)
            else:
                continue
            classes_total_num[x["category_id"]] = classes_total_num[x["category_id"]] + 1

    classes_total = []
    for i in range(0, num_classes):
        classes_total_total[i] = classes_total_h_w[i] / classes_total_num[i]
        classes_total.append([classes_total_total[i], classes_total_max[i]])

    N_COLS = min(3, len(class_names) * 4)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), float(v[0]), float(v[1])] for i, v in enumerate
        (classes_total)])
    )
    # total_num_instances = sum(data[1::2])
    # total_num_instances = 0
    # total_small_instances = 0
    # total_medium_instances = 0
    # total_large_instances = 0
    # for i in range(0, num_classes):
    #     total_num_instances = total_num_instances + data[i * 5 + 1]
    #     total_small_instances = total_small_instances + data[i * 5 + 2]
    #     total_medium_instances = total_medium_instances + data[i * 5 + 3]
    #     total_large_instances = total_large_instances + data[i * 5 + 4]
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    # if num_classes > 1:
    # data.extend(["total", total_num_instances, total_small_instances, total_medium_instances, total_large_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#ratio", "ratio_max"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(table)


def count_instances_class_aspect_ratio_number(dataset_dicts, class_names):
    """
   Args:
       dataset_dicts (list[dict]): list of dataset dicts.
       class_names (list[str]): list of class names (zero-indexed).
   """
    for dataset_name, dicts in zip(class_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    has_instances = "annotations" in dataset_dicts[0]
    if has_instances:
        try:
            _print_instances_class_aspect_ratio_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass


if __name__ == '__main__':
    # 获取类别
    # train_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC"
    # # 统计dotav1 原始数据集的指标
    # image_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val/images"
    # label_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val/labelTxt"
    # 统计dotav2 原始数据集的指标
    train_data_path = '/home/pms5343/python/yolor-main/data/small_33'
    image_data_path = "/home/pms5343/python/yolor-main/data/small_33/images"
    label_data_path = "/home/pms5343/python/yolor-main/data/small_33/labels"
    # 统计切图800后的原始数据集指标
    # image_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/images"
    # label_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/labelTxt"
    # 统计切图600后的原始数据集指标
    # image_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_600/images"
    # label_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_600/labelTxt"
    class_names = ["0"]

    dataset_dicts = load_dota_instances(image_data_path, label_data_path, class_names)
    #print (dataset_dicts)
    
    conunt_instances_class_number(dataset_dicts, class_names)

    count_instances_class_small_middle_large_number(dataset_dicts, class_names)

    #count_instances_class_aspect_ratio_number(dataset_dicts, class_names)