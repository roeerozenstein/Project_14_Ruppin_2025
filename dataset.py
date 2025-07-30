import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm
import os
import pickle
from sklearn.preprocessing import LabelEncoder




def load_h5_directory(directory_path):
    data_list = []

    for file_name in tqdm(os.listdir(directory_path), desc="Reading .h5 files"):
        if file_name.endswith('.h5') or file_name.endswith('.hdf5'):
            file_path = os.path.join(directory_path, file_name)
            try:
                with h5py.File(file_path, 'r') as h5_file:
                    image = h5_file['image'][:]   # (H, W, C)
                    mask = h5_file['mask'][:]     # (H, W)
                    label = h5_file['label'][()]  # לדוגמה: b'T_cell'
                    
                    data_list.append({
                        'file_path': file_path,
                        'image': image,
                        'mask': mask,
                        'label': label
                    })
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    
    return data_list




def pad_crop_to_shape(img, target_size):
    h, w = img.shape[:2]
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_widths = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if img.ndim == 3:
        pad_widths.append((0, 0))

    img = np.pad(img, pad_widths, mode='constant')

    # Crop center if too large
    h, w = img.shape[:2]
    start_y = (h - target_size) // 2
    start_x = (w - target_size) // 2
    return img[start_y:start_y + target_size, start_x:start_x + target_size, ...]



# encodes labels into integers and saves lable to class mapping
def encode_labels(df, label_column='label', output_column='label_encoded', mapping_path='label_mapping.pkl'):
    le = LabelEncoder()

    # Convert bytes to strings if needed
    if isinstance(df[label_column].iloc[0], bytes):
        df[label_column] = df[label_column].apply(lambda x: x.decode('utf-8'))

    # Fit and transform labels
    df[output_column] = le.fit_transform(df[label_column])

    # Create mapping: int -> original label
    label_mapping = dict(zip(le.transform(le.classes_), le.classes_))

    # Save mapping to a pickle file
    with open(mapping_path, 'wb') as f:
        pickle.dump(label_mapping, f)

    print("Label encoding complete. Example mapping:", label_mapping)
    return df, label_mapping



class DataLoader(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx]['image']  # (H, W, 8)
        mask  = self.data.iloc[idx]['mask']   # (H, W, 8)

        # concating the channels
        combined = np.concatenate([image, mask], axis=-1)  # (H, W, 16)
        combined = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1)  # (16, H, W)

        label = self.data.iloc[idx]['label_encoded']
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            combined = self.transform(combined)

        return combined, label