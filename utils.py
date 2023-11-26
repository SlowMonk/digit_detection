import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import h5py

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    
    mat_data = h5py.File(mat_file_path)
    size = mat_data['/digitStruct/name'].size

    for _i in range(size):
    pic = get_name(_i, mat_data)
    box = get_box_data(_i, mat_data)
    print(f'pic:{pic}, box:{box}')
    if _i==2: print(stop)
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

def read_mat_file(mat_file_path):
    with h5py.File(mat_file_path, 'r') as f:
            digitStruct = f['digitStruct']
            print('keys:', digitStruct.keys())
            bbox = digitStruct['bbox']
            names = digitStruct['name']
            print('bbox:', bbox)
            print('names:',names)

            for i in range(len(bbox)):
                try:
                    bbox_entry = bbox[i][0]  # Assuming each entry is a reference to another group or dataset
                    name_entry = names[i][0]
                    bbox_data = f[bbox_entry]
                    name_data = f[name_entry]
                    #print('name_values:', name_data[()])

                    # Accessing the actual data within the 'bbox' entry
                    height_ref = bbox_data['height'][0]
                    width_ref = bbox_data['width'][0]
                    top_ref = bbox_data['top'][0]
                    left_ref = bbox_data['left'][0]
                    name = name_data[0][0]

                    # Extracting the actual height and width values
                    height = f[height_ref[0]][()].item()
                    width = f[width_ref[0]][()].item()
                    top = f[top_ref[0]][()].item()
                    left = f[left_ref[0]][()].item()

                    print(f'height:{height}, width:{width}, top:{top}, left:{left}, name:{name}')
                    print(stop)
                except:
                    print(stop)
                    continue
                
def visualize_sample_dataset(custom_dataset, idx):
    # Load the image using PIL
    image_path = custom_dataset[idx][0]['name']  # Assuming the list contains dictionaries
    image = Image.open(image_path)
    print('image:', np.array(image).shape)
    # Visualize the image
    #image = image.resize((32, 32))

    plt.imshow(image)

    # Visualize each bounding box
    for box_info in custom_dataset[idx]:
        # Access bounding box coordinates
        bbox = (box_info['left'], box_info['top'], box_info['width'], box_info['height'])

        # Create a Rectangle patch
        #rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')\
        rect = Rectangle((bbox[0], image.height - bbox[1] - bbox[3]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        plt.gca().add_patch(rect)

    # Show the plot
    plt.show()              
    
def visualize_sample(sample):
    # Load the image using PIL
    image_path = sample['name']
    image = Image.open(image_path)

    # Visualize the image
    plt.imshow(image)

    # Access bounding box coordinates
    for i in range(len(sample['height'])):
        bbox = (sample['left'][i], sample['top'][i], sample['width'][i], sample['height'][i])

        # Create a Rectangle patch
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        plt.gca().add_patch(rect)

    # Show the plot
    plt.show()
    

def detect_digit_from_image(img_path, model, output_image_path, device):
    # Load your model
    #model =  md(num_classes=11).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Load and preprocess the image
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the window size and step size
    window_size = (32, 32)
    step_size = 10

    # List to store bounding box coordinates
    bbox_coords = []

    # Sliding window
    for y in range(0, gray.shape[0] - window_size[1], step_size):
        for x in range(0, gray.shape[1] - window_size[0], step_size):
            # Extract and preprocess the window
            window = gray[y:y + window_size[1], x:x + window_size[0]]
            processed_window = cv2.resize(window, (32, 32))
            processed_window = transforms.ToTensor()(processed_window)
            processed_window = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(processed_window)
            processed_window = processed_window.unsqueeze(0)

            # Classify the window
            with torch.no_grad():
                outputs = model(processed_window)
                _, predicted = torch.max(outputs, 1)

            # If it's a digit, save the bounding box coordinates
            if predicted.item() > 0:
                bbox_coords.append((x, y, x + window_size[0], y + window_size[1]))

    # Draw bounding boxes on the original image
    for (x1, y1, x2, y2) in bbox_coords:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the final image with bounding boxes
    cv2.imwrite(output_image_path, image)



def vae_loss(reconstructed_x, x, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence