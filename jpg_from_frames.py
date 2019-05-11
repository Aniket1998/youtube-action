import os
import numpy as np
import torch
import imageio
from PIL import Image


def create_vids(vidpath, crop_size=128, cut_frames=20, skip_frames=4, threshold=16):
    vid = imageio.get_reader(vidpath, 'ffmpeg')
    listframes = []
    frame_count = 0
    for frame in vid:
        if frame_count % skip_frames == 0:
            frame_img = Image.fromarray(frame)
            frame_img = frame_img.resize((crop_size, crop_size), resample=Image.LANCZOS)
            listframes.append(np.array(frame_img))
        frame_count += 1
    nframes = len(listframes)
    vid_list = []
    reject = 0
    for i in range(nframes // cut_frames + 1):
        slice_frame = listframes[i * cut_frames: (i+1)*cut_frames]
        if len(slice_frame) == cut_frames:
            vid_list.append(slice_frame)
        elif len(slice_frame) >= threshold:
            last_frame = slice_frame[-1]
            while len(slice_frame) < cut_frames:
                slice_frame.append(last_frame)
            vid_list.append(slice_frame)
        else:
            reject +=1
    vid_list = np.array(vid_list)
    if vid_list.shape[0] != 0:
        vid_list = vid_list.reshape(vid_list.shape[0], vid_list.shape[1] * vid_list.shape[2],  vid_list.shape[3], vid_list.shape[4])
    return vid_list, reject



action_categories = os.listdir('./raw')
test_accept_rate = 0.15

num_train = 0
num_test = 0
exclusion_count = 0

train_classes = [0 for _ in range(len(action_categories))]
test_classes = [0 for _ in range(len(action_categories))]
for action_label in range(len(action_categories)):
    print('Preparing videos for action category {}'.format(action_categories[action_label]))
    for subcat in os.listdir('./raw/' + action_categories[action_label]):
        for video in os.listdir('./raw/' + action_categories[action_label] + '/' + subcat):
            vid_list, e = create_vids('./raw/' + action_categories[action_label] + '/' + subcat + '/' + video)
            exclusion_count += e
            if vid_list.shape[0] != 0:
                for vid_frames in vid_list:
                    img = Image.fromarray(vid_frames)
                    if np.random.rand() <= test_accept_rate:
                        img.save('./processed_data/test/{}-{}.jpeg'.format(action_categories[action_label], test_classes[action_label]))
                        num_test += 1
                        test_classes[action_label] += 1
                    else:
                        img.save('./processed_data/train/{}-{}.jpeg'.format(action_categories[action_label], test_classes[action_label]))
                        num_train += 1
                        train_classes[action_label] += 1

 
print('Training Set', num_train)
print('Test Set', num_test)
print('Excluded', exclusion_count)

print('Training Set Categories')
for i in range(len(action_categories)):
    print('{} : {}'.format(action_categories[i], train_classes[i]))

print('Test Set Categories')
for i in range(len(action_categories)):
    print('{} : {}'.format(action_categories[i], test_classes[i]))
