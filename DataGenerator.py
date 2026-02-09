import numpy as np
import scipy.io as spio
import random
import pickle
import math
import os
import timeit
import collections
from multiprocessing import Pool, cpu_count

from torch.utils.data import Dataset
from PIL import Image

class CifarDataGenerator(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x))
    
    def __getitem__(self, index):
        
        img, target = self.x[index,:,:,:], self.y[index]

        img = img.transpose((2, 0, 1))
        target = target[0]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target

class MnistDataGenerator(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x))
    
    def __getitem__(self, index):
        
        img, target = self.x[index,:], self.y[index]

        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target

class MixtureDataGenerator(Dataset):
    def __init__(self, x, y, transform=None, scale=1, trigger=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.scale = scale
        self.trigger = trigger
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x)*self.scale)
    
    def __getitem__(self, index):
        
        img, target = self.x[index,:,:,:], self.y[index]

        img = img.transpose((2, 0, 1))
        
        if not self.trigger:
            target = target[0] # for cifar
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target

class IQDataGenerator(Dataset):
    def __init__(self, x,y, transform = None, convert_to_spectrogram=False):
        self.x = x
        self.y = y
        # print(self.x)
        self.transform = transform
        self.convert_to_spectrogram = convert_to_spectrogram

    # def __getdata__(self):


    def __len__(self):
        """Return total number of samples."""
        # print(self.x.shape, len(self.x))
        return self.x.shape[0]

    def __getitem__(self, index):
        """Retrieve one sample and apply transformation."""
        iq_sample = self.x[index,:]  # This should be interleaved [I1, Q1, I2, Q2, ...]

        # Convert from interleaved [I1, Q1, I2, Q2, ...] to [I, Q] format
        iq_sample = np.stack([iq_sample.real, iq_sample.imag], axis=0).astype(np.float32) 

        label = self.y[index]

        # Optionally convert to spectrogram
        if self.convert_to_spectrogram:
            iq_sample = self._convert_to_spectrogram(iq_sample)

        # Apply PyTorch transform if specified (e.g., normalization)
        if self.transform:
            iq_sample = self.transform(iq_sample)

        return iq_sample, label

    def _convert_to_spectrogram(self, iq_sample):
        """Convert IQ samples to spectrogram using STFT."""
        f, t, Sxx = spectrogram(iq_sample, nperseg=64)  # Compute spectrogram
        return np.log10(Sxx + 1e-12)  # Log-scale for better numerical stability
