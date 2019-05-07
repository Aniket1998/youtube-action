import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import imageio
from PIL import Image


def create_tensors(vidpath, action_label, crop_size=128, cut_frames=20):
    vid = imageio.get_reader(vidpath, 'ffmpeg')
    transforms = T.Compose([T.ToPILImage(), T.Resize((crop_size, crop_size)), T.ToTensor()])
    listframes = []
    for frame in vid:
        listframes.append(transforms(frame))
    nframes = len(listframes)
    listframes = torch.stack(listframes)
    tensor_list = []
    reject = 0
    for i in range(nframes // cut_frames + 1):
        slice_frame = listframes[i * cut_frames: (i+1)*cut_frames]
        if slice_frame.size(0) == cut_frames:
            tensor_dict = {
                    'frames': slice_frame,
                    'label': np.array(action_label)
            }
            tensor_list.append(tensor_dict)
        else:
            reject +=1
    return tensor_list, reject



action_categories = os.listdir('./raw')
test_accept_rate = 0.20

num_train = 0
num_test = 0
exclusion_count = 0

train_classes = [0 for _ in range(len(action_categories))]
test_classes = [0 for _ in range(len(action_categories))]
for action_label in range(len(action_categories)):
    print('Preparing videos for action category {}'.format(action_categories[action_label]))
    for subcat in os.listdir('./raw/' + action_categories[action_label]):
        for video in os.listdir('./raw/' + action_categories[action_label] + '/' + subcat):
            tensor_list, e = create_tensors('./raw/' + action_categories[action_label] + '/' + subcat + '/' + video, action_label)
            exclusion_count += e
            for tensor_dict in tensor_list:
                if np.random.rand() <= test_accept_rate:
                    torch.save(tensor_dict, './large_tensor_data/test/{}.vid'.format(num_test))
                    num_test += 1
                    test_classes[action_label] += 1
            else:
                torch.save(tensor_dict, './large_tensor_data/train/{}.vid'.format(num_train))
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
