"""
File for data preprocessing
Generating the csv files storing the path of the training set, validation set and test set
"""
import pandas as pd
import csv

def generate_csv():
    read_train = open("./dataset/train/train.txt", "r").readlines()

    with open("train.csv", mode="w", newline="") as train_file:
        writer = csv.writer(train_file)
        writer.writerow(("img", "text"))
        for line in read_train:
            image_file = line.split(" ")[0]
            text_file = image_file.replace(".png", ".txt")
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)


    read_train = open("./dataset/test/test.txt", "r").readlines()

    with open("test.csv", mode="w", newline="") as train_file:
        writer = csv.writer(train_file)
        writer.writerow(("img", "text"))
        for line in read_train:
            image_file = line.split(" ")[0]
            text_file = image_file.replace(".png", ".txt")
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)

    return


def train_val_split(val_ratio, seed):

    data = pd.read_csv("train.csv")

    data_val = data.sample(frac=val_ratio, random_state=seed, axis=0)
    data_train = data.drop(data_val.index)

    data_val.to_csv("val.csv", index=False)
    data_train.to_csv("train.csv", index=False)