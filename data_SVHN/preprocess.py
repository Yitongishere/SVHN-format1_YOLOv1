"""
File for data preprocessing
Load the label data from the original .mat files and store them into txt files
Converting the labels format between "corner" and "midpoint"
"""
import numpy as np
import h5py
from pix_to_ratio import pix_to_ratio
from generate_csv_svhn import generate_csv, train_val_split
import pandas as pd

def get_name(index, hdf5_data):

    name_ref = hdf5_data['digitStruct']['name'][index].item()
    name = ''.join([chr(v[0]) for v in hdf5_data[name_ref]])

    return name


def get_bbox(index, hdf5_data):

    attrs = {}
    bbox_ref = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[bbox_ref][key]
        values = [hdf5_data[attr[i].item()][0][0].astype(int) \
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values

    return attrs


def ltwh_to_ltrb(bbox_ltwh):

    bbox_ltrb = {}
    bbox_ltrb['label'] = (np.array(bbox_ltwh['label']) % 10).tolist()
    bbox_ltrb['left'] = bbox_ltwh['left']
    bbox_ltrb['top'] = bbox_ltwh['top']
    bbox_ltrb['right'] = np.add(bbox_ltwh['left'], bbox_ltwh['width']).tolist()
    bbox_ltrb['bottom'] = np.add(bbox_ltwh['top'], bbox_ltwh['height']).tolist()

    return bbox_ltrb

def ltwh_to_xywh(bbox_ltwh):

    bbox_xywh = {}
    bbox_xywh['label'] = (np.array(bbox_ltwh['label']) % 10).tolist()
    bbox_xywh['mid_x'] = np.add(bbox_ltwh['left'], 0.5 * np.array(bbox_ltwh['width'])).tolist()
    bbox_xywh['mid_y'] = np.add(bbox_ltwh['top'], 0.5 * np.array(bbox_ltwh['height'])).tolist()
    bbox_xywh['width'] = bbox_ltwh['width']
    bbox_xywh['height'] = bbox_ltwh['height']

    return bbox_xywh


def mat_to_txt(txtfile, hdf5_data, format):
    with open(txtfile, 'w') as file:
        for i in range(len(hdf5_data['digitStruct']['name'])):
            bbox = get_bbox(i, hdf5_data)

            if format == 'original':
                bbox_str = ''
                bbox_str += get_name(i, hdf5_data)
                bbox_str += ' '
                for j in range(len(bbox['label'])):
                    bbox_str += ''.join(str(bbox['label'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['left'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['top'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['width'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['height'][j]))
                    bbox_str += ' '

            if format == 'corner':
                bbox = ltwh_to_ltrb(bbox)
                bbox_str = ''
                bbox_str += get_name(i, hdf5_data)
                bbox_str += ' '
                for j in range(len(bbox['label'])):
                    bbox_str += ''.join(str(bbox['label'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['left'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['top'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['right'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['bottom'][j]))
                    bbox_str += ' '

            if format == 'midpoint':
                bbox = ltwh_to_xywh(bbox)
                bbox_str = ''
                bbox_str += get_name(i, hdf5_data)
                bbox_str += ' '
                for j in range(len(bbox['label'])):
                    bbox_str += ''.join(str(bbox['label'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['mid_x'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['mid_y'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['width'][j]))
                    bbox_str += ','
                    bbox_str += ''.join(str(bbox['height'][j]))
                    bbox_str += ' '

            file.write(bbox_str[:-1])
            file.write('\n')
    return


def txt_to_txts(txt_path, dir_path):

    ori_file = open(txt_path, "r").readlines()

    for line in ori_file:
        name = line.split(".")[0]
        txt_name = dir_path + name + ".txt"
        with open(txt_name, mode="w", newline="") as new_file:
            label = line.split(" ")[1:]
            for i in range(len(label)):
                new_file.write(label[i])
                new_file.write('\n')

    return





if __name__ == "__main__":

    train_path = './dataset/train/digitStruct.mat'
    test_path = './dataset/test/digitStruct.mat'

    hdf5_data_train = h5py.File(train_path, 'r')
    hdf5_data_test = h5py.File(test_path, 'r')

    print("Converting .mat file to .txt file ...")
    mat_to_txt('./dataset/train/train.txt', hdf5_data_train, 'midpoint')
    mat_to_txt('./dataset/test/test.txt', hdf5_data_test, 'midpoint')

    print("Generating .txt files for each image ...")
    txt_to_txts('./dataset/train/train.txt', "./dataset/train_labels/")
    txt_to_txts('./dataset/test/test.txt', "./dataset/test_labels/")

    print("Converting pixel-wise to ratio-wise labels ...")
    label_file_train = open('./dataset/train/train.txt', "r").readlines()
    for i in range(len(label_file_train)):
        pix_to_ratio(str(i+1), 'train')

    label_file_test = open('./dataset/test/test.txt', "r").readlines()
    for i in range(len(label_file_test)):
        pix_to_ratio(str(i+1), 'test')

    print("Generating csv files ...")
    generate_csv()

    print("Seperating validation set from the Whole training set ...")
    train_val_split(val_ratio=0.1, seed=42)

    # I found a flawed data 3192.png & 3192.txt and delete it from the testset,
    # as in this example, the midpoint of the bbox is labeled outside the image
    test_csv = pd.read_csv("test.csv")
    new_test_csv = test_csv.drop([3192-1])
    new_test_csv.to_csv("test.csv", index=False)









