"""
The function allows you convert the pixel-wise label to ratio-wise label in terms of the corresponding image
"""
from PIL import Image

def pix_to_ratio(pre_name, train_or_test):
    image_path = "./dataset/" + train_or_test + "/" + pre_name + ".png"
    label_path = "./dataset/" + train_or_test + "_labels/" + pre_name + ".txt"
    new_label_path = "./dataset/" + train_or_test + "_labels_ratio/" + pre_name + ".txt"

    img_file = Image.open(image_path)
    img_size = img_file.size
    label_file = open(label_path, "r").readlines()

    with open(new_label_path, mode="w", newline="") as new_file:
        for line in label_file:
            if line == '\n':
                break
            new_str = ''
            c = line.split(",")[0]
            c = int(float(c))
            mid_x = line.split(",")[1]
            mid_x = float(mid_x) / img_size[0]
            mid_y = line.split(",")[2]
            mid_y = float(mid_y) / img_size[1]
            w = line.split(",")[3]
            w = float(w) / img_size[0]
            h = line.split(",")[4]
            h = float(h) / img_size[1]

            new_str += str(c)
            new_str += str(' ')
            new_str += str(mid_x)
            new_str += str(' ')
            new_str += str(mid_y)
            new_str += str(' ')
            new_str += str(w)
            new_str += str(' ')
            new_str += str(h)

            new_file.write(new_str)
            if line == label_file[-1]:
                break
            new_file.write('\n')
        new_file.close()




if __name__ == "__main__":
    pre_name = '1'
    pix_to_ratio(pre_name, 'train')
