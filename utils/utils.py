import torch
import yaml
import csv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

def read_tiff(tiff_file_name: str):
  img = Image.open(tiff_file_name)
  return torch.from_numpy(np.array(img)).float()


def _read_picture(img_file_name: str):
  img = cv2.imread(img_file_name, cv2.IMREAD_UNCHANGED)
  return torch.from_numpy(img).float()


def read_png(png_file_name: str):
  return _read_picture(png_file_name)


def read_jpeg(jpeg_file_name: str):
  return _read_picture(jpeg_file_name)


def read_npy(npy_file_name: str):
  array = np.load(npy_file_name)
  return torch.from_numpy(array).float()


def load_config(yaml_file_name: str):
  with open(yaml_file_name, 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config


def load_psf(psf_tiff: str, psf_ratio: int = 1):
  psf = read_tiff(psf_tiff)
  if psf_ratio > 1:
    psf = psf.permute(2, 0, 1)
    psf = torch.nn.functional.avg_pool2d(psf, kernel_size=psf_ratio)
    psf = psf.permute(1, 2, 0)
  psf /= psf.max()
  return psf


def load_imglist(csv_file_name: str):
  with open(csv_file_name, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    imglist = [row[0] for row in reader]
    return imglist


def to_numpy(x):
  return x.detach().cpu().numpy()
