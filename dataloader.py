import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import h5py
import os

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and label as a dictionary or tuple
        sample = {'image': image, 'label': label}

        return sample

class SVHDDataset(Dataset):
    
    def __init__(self, mat_file_path, image_dir, mode):
        with h5py.File(mat_file_path, 'r') as f:
            if mode=='train':
                print('## train ##')
                # Training dataset structure
                self.digitStruct = f['digitStruct']
                self.bbox_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['bbox'])]
                self.name_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['name'])]
            else:
                print('## test ##')
                # Test dataset structure
                self.digitStruct = f
                self.bbox_refs = [f[key][()] for key in f['bbox'].keys()]
                self.name_refs = [f[key][()] for key in f['name'].keys()]

            self.length = len(self.digitStruct)

        self.file = h5py.File(mat_file_path, 'r')  # Open the file separately to keep it open
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #try:
        # Access image and label information from bbox and name
        name_data = self.file[self.name_refs[idx]]
        bbox_data = self.file[self.bbox_refs[idx]]

        num_boxes = bbox_data['height'].shape[0]

        samples = []

        for i in range(num_boxes):
            height_ref = bbox_data['height'][i]
            width_ref = bbox_data['width'][i]
            top_ref = bbox_data['top'][i]
            left_ref = bbox_data['left'][i]
            label_ref = bbox_data['label'][i]
        
            try:
                height = self.file[height_ref[0]][()].item()
                width = self.file[width_ref[0]][()].item()
                top = self.file[top_ref[0]][()].item()
                left = self.file[left_ref[0]][()].item()
                label = self.file[label_ref[0]][()].item()
            except:
                height = height_ref[0]
                width = width_ref[0]
                top = top_ref[0]
                left = left_ref[0]
                label = label_ref[0]

            # Load the image using PIL
            image_path = os.path.join(self.image_dir , str(idx+1) + f'.png') # assuming the name is a byte string
            image = Image.open(image_path)

            # Crop the image based on the bounding box
            cropped_image = image.crop((left, top, left + width, top + height))

            # Apply the specified transformations
            cropped_image = self.transform(cropped_image)


            # /Users/jakec/Desktop/svhd_dataset/train
            name = os.path.join( self.image_dir , str(idx+1) + f'.png')

            sample = {
                'image': cropped_image,
                'label': label,
                'name': name
            }
            samples.append(sample)

        return samples
    
class SVHDDigitOnlyDigitDataset(Dataset):
    
    def __init__(self, mat_file_path, image_dir, mode):
        # 파일 열기 및 데이터셋 구조 읽기
        with h5py.File(mat_file_path, 'r') as f:
            if mode == 'train':
                print('## train ##')
                self.digitStruct = f['digitStruct']
                self.bbox_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['bbox'])]
                self.name_refs = [obj_ref[0] for _, obj_ref in enumerate(self.digitStruct['name'])]
            else:
                print('## test ##')
                self.digitStruct = f
                self.bbox_refs = [f[key][()] for key in f['bbox'].keys()]
                self.name_refs = [f[key][()] for key in f['name'].keys()]

            self.length = len(self.digitStruct)

        self.file = h5py.File(mat_file_path, 'r')  # 파일 따로 열기
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4342, 0.4431, 0.4768], std=[0.1927, 0.1956, 0.1931]),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 이미지 및 레이블 정보 액세스
        bbox_data = self.file[self.bbox_refs[idx]]

        num_boxes = bbox_data['height'].shape[0]

        digit_samples = []
        non_digit_samples = []

        for i in (range(num_boxes)):
            # bbox 정보 로드
            height_ref = bbox_data['height'][i]
            width_ref = bbox_data['width'][i]
            top_ref = bbox_data['top'][i]
            left_ref = bbox_data['left'][i]
            label_ref = bbox_data['label'][i]

            try:
                height = self.file[height_ref[0]][()].item()
                width = self.file[width_ref[0]][()].item()
                top = self.file[top_ref[0]][()].item()
                left = self.file[left_ref[0]][()].item()
                label = self.file[label_ref[0]][()].item()
            except:
                height = height_ref[0]
                width = width_ref[0]
                top = top_ref[0]
                left = left_ref[0]
                label = label_ref[0]

            # 이미지 로드 및 크롭
            image_path = os.path.join(self.image_dir, str(idx + 1) + '.png')
            image = Image.open(image_path)
            cropped_image = image.crop((left, top, left + width, top + height))
            cropped_image = self.transform(cropped_image)

            digit_sample = {
                'image': cropped_image,
                'label': 1,
                'name': image_path
            }
            digit_samples.append(digit_sample)
    
        # Digit 및 Non-Digit 샘플 결합
        return digit_samples


def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][:]])