import torch
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transition_network import TransitionCNN
import os
import shutil
import time
from TestVideo import TestVideo, return_start_and_end
from video_processing import six_four_crop_video
import cv2

# first decompose the video to frames
# place the video to be detected into the directory 
#
video = 'sny.mp4'

text_file = 'frames.txt'

print('Decomposing video to frames .....')


frames_path = 'video_frames/'
os.makedirs('video_frames/', exist_ok=True)

videoCapture = cv2.VideoCapture()
videoCapture.open(video)
frame_num = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
f = open(text_file, 'w+')
for i in range(int(frame_num)):
    _,frame = videoCapture.read()
    frame = six_four_crop_video(frame)          #resize and crop to 64*64
    frame_path = frames_path + 'frame_' + str(i + 1) + '.png'
    cv2.imwrite(frame_path,frame)
    f.write(frame_path + '\n')
f.close()



print('Frame decomposition complete! ')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#load model
model = TransitionCNN()
model.load_state_dict(torch.load('./finetune_models_42/shot_boundary_detector_even_distrib.pt'))
model.to(device)

prediction_text_file  = 'result_42.txt'

pred_file = open(prediction_text_file, 'w+')
pred_file.write(str(1)+' ')


print('Computing predictions for video', video, '...................' )
start_time = time.time()  # start count


test_video = TestVideo('frames.txt', sample_size=100, overlap=9)
test_loader = DataLoader(test_video, batch_size=1, num_workers=3)  #get test data

video_indexes = []
vals = np.arange(test_video.get_line_number())
length = len(test_video)

for val in range(length):
    s,e = return_start_and_end(val, length = len(vals))
    video_indexes.append(vals[s:e])

for indx, batch in enumerate(test_loader):
        batch.to(device)
        batch = batch.type('torch.cuda.FloatTensor') if device == 'cuda' else batch.type('torch.FloatTensor')
        predictions = model(batch.to(device))
        predictions = predictions.argmax(dim=1).cpu().numpy()
        for idx, prediction_set in enumerate(predictions):
            for i, prediction in enumerate(prediction_set):
                if prediction[0][0] == 0:
                    frame_index = video_indexes[indx][i+5]
                    pred_file.write(str(frame_index) + '\n')
                    pred_file.write(str(frame_index+1)+' ')  #write result to txt

pred_file.write(str(len(vals))) #last frame
pred_file.close()

# delete files used for process
os.remove('frames.txt')
shutil.rmtree('video_frames/')

end_time = time.time()
print('run time:',str(end_time - start_time),'ms')
print('Predictions complete !!!')













