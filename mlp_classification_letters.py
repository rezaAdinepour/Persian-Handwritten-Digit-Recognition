"""
This program takes a dataset of images, convert them to 8*8 size
and trains an MLP net from mlp module.
It represents loss over train and test dataset in each 10 epochs.
"""

import numpy as np
import torch
from torch.nn.functional import one_hot
from cv2 import imread, IMREAD_GRAYSCALE
import glob
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import mlp


# this function gets block mean of original images and returns size*size arrays containing block mean values
def resize(images, size=8):
    resized = []
    for image in images:
        row_num = image.shape[0]
        col_num = image.shape[1]

        length = torch.linspace(0, row_num, size+1).data.numpy()
        width = torch.linspace(0, col_num, size+1).data.numpy()

        num = [0 for _ in range(size)]
        rows8 = [[0 for _ in range(col_num)] for j in range(size)]
        for i in range(row_num):
            for l in range(1, size+1):
                if i <= length[l]:
                    temp = np.array(rows8[l - 1]) * num[l - 1]
                    rows8[l - 1] = np.add(temp, image[i]) / (num[l - 1] + 1)
                    num[l - 1] += 1
                    break

        num = [0 for _ in range(size)]
        data8 = [[0 for _ in range(size)] for j in range(size)]
        for i in range(col_num):
            for w in range(1, size+1):
                if i <= width[w]:
                    temp = np.array(data8[w - 1]) * num[w - 1]
                    data8[w - 1] = np.add(temp, np.array(rows8).transpose()[i]) / (num[w - 1] + 1)
                    num[w - 1] += 1
                    break

        data8 = np.array(data8).transpose()
        resized.append(np.array(data8))
    return np.array(resized)


def present(epoch, train_loss, test_loss):
    x = range(10, epoch+2, 10)
    plt.cla()
    plt.plot(x, train_loss, color='blue', label='train data', alpha=0.5, marker='o')
    plt.plot(x, test_loss, color='red', label='test data', alpha=0.5, marker='o')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.pause(0.01)


def main():
    max_epoch = 1500
    learning_rate = 0.1
    batch_size = 5

    path = './bmp/*.bmp'
    files = []
    files.extend(glob.glob(path))
    labels = [int(file.split('_')[0][-1]) for file in files]
    paths_train, paths_test, labels_train, labels_test = train_test_split(
        files, labels, test_size=0.4)

    # expanding labels
    labels_train_expanded = one_hot(torch.tensor(labels_train))
    labels_test_expanded = one_hot(torch.tensor(labels_test))

    # reading images
    images_train = [imread(file, IMREAD_GRAYSCALE) for file in paths_train]
    images_test = [imread(file, IMREAD_GRAYSCALE) for file in paths_test]

    # resizing pictures
    ims_train_64 = resize(images_train)
    ims_test_64 = resize(images_test)

    # flattening images
    ims_train_flt = ims_train_64.flatten().reshape((ims_train_64.shape[0], 64))
    ims_test_flt = ims_test_64.flatten().reshape((ims_test_64.shape[0], 64))

    # normalizing each image
    im_train_norm = ims_train_flt/255
    im_test_norm = ims_test_flt/255

    net = mlp.MLP(learning_rate, batch_size, [64, 20, 10])
    train_loss = []
    test_loss = []
    for epoch in range(max_epoch):
        net.train(im_train_norm, labels_train_expanded)
        if epoch % 10 == 9:
            prediction, loss = net.predict(im_train_norm, labels_train_expanded)
            train_loss.append(loss)
            prediction, loss = net.predict(im_test_norm, labels_test_expanded)
            test_loss.append(loss)

            present(epoch, train_loss, test_loss)


if __name__ == '__main__':
    main()
