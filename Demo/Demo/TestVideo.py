# data set object for inference of video for shot boundary detection
# videos are processed in batches of 100 frames with an overlap of 9 frames

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from snippet import getSnippet
from math import floor
from transition_network import TransitionCNN
from utilities import normalize_frame, print_shape
import pandas as pd
import os
import time

def return_start_and_end(idx, length, sample_size=100, overlap=9):
    if idx == 0:
        start = 0
        end = start + sample_size
    else:
        start = idx*(sample_size - overlap)
        end = start + sample_size
        if end>(length-1):
            end = length-1
    return start, end


class TestVideo(Dataset):

    def __init__(self, video_file, sample_size=100, overlap=9):
        # video file is text file  path with all frame listings
        with open(video_file) as f:
            lines = f.readlines()

        self.lines = [line.strip() for line in lines]  #remove /n
        self.line_number = len(self.lines)
        self.sample_size = sample_size
        self.overlap = overlap

    def __len__(self):
        return floor(self.line_number/(self.sample_size - self.overlap))

    def __getitem__(self, idx):
        start, end = return_start_and_end(idx=idx, length = self.line_number, sample_size=self.sample_size, overlap=self.overlap)
        video_snippet = np.array(getSnippet(self.lines, start, end))
        #transpose the individual frames to the in the correct format and the fully returned structure
        video_snippet = np.array([normalize_frame(frame) for frame in video_snippet])
        video_snippet = np.array([np.transpose(frame, (2,0,1)) for frame in video_snippet])
        video_snippet = np.transpose(video_snippet, (1,0,2,3))
        return video_snippet

    def get_line_number(self):
        return self.line_number