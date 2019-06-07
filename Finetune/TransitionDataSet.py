import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Transitiondataset(Dataset):
    def __init__(self, imgs_path, label_path):

        f_img = open(imgs_path, 'r')
        count = int(os.popen('wc -l '+imgs_path).read().split()[0])
        self.imgs = np.empty((count, 64, 64, 3), dtype='float32')
        i = 0
        for line in f_img:
            item = line.rstrip()
            img = cv2.imread(item)
            arr = np.asarray(img, dtype='float32')
            self.imgs[i,:,:,:]=(arr/255.0- 0.45)/0.225
            i += 1
        f_img.close()

        f_label = open(label_path, 'r')
        self.label = np.ones(count,dtype='int')
        for line in f_label:
            word = line.split()
            num = int(word[1])
            self.label[num]=0
        f_label.close()

        self.all_imgs = np.empty((count-9, 10, 64, 64, 3), dtype='float32')
        self.all_labels = np.ones(count-9, dtype='int')
        for i in range(count-9):
            for j in range(10):
                self.all_imgs[i,j,:,:,:] = self.imgs[i+j,:,:,:]
            if self.label[i+4]==0:
                self.all_labels[i] = 0 # is shot boundary


        self.all_imgs = torch.from_numpy(self.all_imgs)
        self.all_labels = torch.from_numpy(self.all_labels)

    def __getitem__(self, index):
        return self.all_imgs[index,:,:,:,:], self.all_labels[index]

    def __len__(self):
        return (len(self.all_labels))

# train_data = Transitiondataset('./train_file/train_1.txt','./ground_truths/train_1.txt')
# train_loader = DataLoader(train_data, batch_size=10, num_workers=2)
# for index, batch in enumerate(train_loader):
#     print(index, batch[1])