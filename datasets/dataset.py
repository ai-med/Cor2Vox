import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision import transforms as T, utils
from torch import nn, einsum
import torch.nn.functional as F
import h5py

from Register import Registers
from PIL import Image
import os
import numpy as np
import torchio as tio
import monai.transforms as montrans
import nibabel as nib
import logging
from tqdm import tqdm

LOG = logging.getLogger(__name__)

from datasets.dataset_utils import pair_file, read_image, resize_img

def create_cortex_sdf(sdf1, sdf2):
    # Create a mask where signs are the same
    same_sign_mask = np.sign(sdf1) == np.sign(sdf2)

    # Initialize the combined SDF
    sdf_combine = np.zeros_like(sdf1)

    # For positions where both are negative (inside both surfaces)
    inside_mask = (sdf1 < 0) & (sdf2 < 0)
    sdf_combine[inside_mask] = np.maximum(sdf1[inside_mask], sdf2[inside_mask])

    # For positions where both are positive (outside both surfaces)
    outside_mask = (sdf1 > 0) & (sdf2 > 0)
    sdf_combine[outside_mask] = np.minimum(sdf1[outside_mask], sdf2[outside_mask])

    # For positions where signs differ, set to zero or handle accordingly
    differing_sign_mask = ~same_sign_mask
    sdf_combine[differing_sign_mask] = 0  # Or handle differently if needed

    return sdf_combine


@Registers.datasets.register_with_name('c2v')
class C2vDataset(Dataset):
    def __init__(
            self,
            dataset_config,
            transform=None,
            target_transform=None,
            stage = 'train',
        ):
        super().__init__()

        folder_suffix = {
            'train': 'Tr',
            'val': 'Val',
            'test': 'Ts',
        }

        if stage not in folder_suffix:
            raise NotImplementedError(f"Stage '{stage}' is not supported.")

        # Construct input and target folders
        self.input_folder = dataset_config.shape_folder + folder_suffix[stage]
        self.target_folder = dataset_config.img_folder + folder_suffix[stage]

        # Include the second input folder if provided
        self.input_folder = [self.input_folder]
        if dataset_config.shape_folder_2 is not None:
            self.input_folder.append(os.path.join(dataset_config.shape_folder_2 + folder_suffix[stage]))

        # if there exist dataset_config.condition_folder, then add it
        if hasattr(dataset_config, 'condition_folder'):
            self.condition_folder = dataset_config.condition_folder + folder_suffix[stage]
            self.condition_folder = [self.condition_folder]
            if hasattr(dataset_config, 'condition_folder_2'):
                self.condition_folder.append(dataset_config.condition_folder_2 + folder_suffix[stage])
            if hasattr(dataset_config, 'condition_folder_3'):
                self.condition_folder.append(dataset_config.condition_folder_3 + folder_suffix[stage])
            if hasattr(dataset_config, 'condition_folder_4'):
                self.condition_folder.append(dataset_config.condition_folder_4 + folder_suffix[stage])
            print('condition_folder: ', self.condition_folder)
        else:
            self.condition_folder = None
        

        self.pair_files = pair_file(self.input_folder, self.target_folder, self.condition_folder)
        
        self.input_size = dataset_config.input_size
        self.depth_size = dataset_config.depth_size
        self.scaler = tio.RescaleIntensity(out_min_max=(0, 1))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        all_data = self.pair_files[index]

        input_files = all_data[0]
        target_file = all_data[1]
        scan_id = all_data[-1]

        # Process all input files in the list
        input_imgs = []
        for input_file in input_files:
            input_img = read_image(input_file, self.scaler, pass_scaler=False)
            input_img = resize_img(input_img, self.input_size, self.input_size, self.depth_size)
            input_imgs.append(input_img)
        
        input_img = np.stack(input_imgs, axis=0)

        input_img = create_cortex_sdf(input_img[0], input_img[1])[np.newaxis, ...]

        target_img = read_image(target_file, self.scaler)
        target_img = resize_img(target_img, self.input_size, self.input_size, self.depth_size)

        transform = Compose([
            Lambda(lambda t: torch.tensor(t).float()),
            Lambda(lambda t: (t * 2) - 1),
            Lambda(lambda t: t.unsqueeze(0)),
        ])


        input_img = torch.cat([transform(img) for img in input_imgs], dim=0)
        target_img = transform(target_img)

        if len(all_data) > 3 and self.condition_folder is not None:
            condition_files = all_data[2]
            condition_imgs = []
            for condition_file in condition_files:
                condition_img = read_image(condition_file, self.scaler, pass_scaler=False)
                condition_img = resize_img(condition_img, self.input_size, self.input_size, self.depth_size)
                condition_imgs.append(condition_img)
            condition_img = np.stack(condition_imgs, axis=0)
            condition_img = torch.cat([transform(img) for img in condition_img], dim=0)
            return (target_img, 'img'), (input_img, 'sdf'), (condition_img, 'condition'), scan_id
        
        return (target_img, 'img'), (input_img, 'sdf'), scan_id
