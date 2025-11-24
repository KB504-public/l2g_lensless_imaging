import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2, rfft2, irfft2, fftshift, ifftshift
from typing import Literal

def non_neg(x):
  return torch.clamp(x, min=0)


def pad_last_two_dims(x, pad_shape, mode: Literal['constant', 'replicate'] = 'constant'):
  assert(len(pad_shape) == 2)
  is_squeeze_dims = False
  if len(x.shape) == 2:
    x = x.unsqueeze(0)
    is_squeeze_dims = True
  meas_shape = (x.shape[-2], x.shape[-1])
  pad_left = (pad_shape[1] - meas_shape[1]) // 2
  pad_right = pad_shape[1] - meas_shape[1] - pad_left
  pad_top = (pad_shape[0] - meas_shape[0]) // 2
  pad_bottom = pad_shape[0] - meas_shape[0] - pad_top
  result = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
  if is_squeeze_dims:
    result = result.squeeze(0)
  return result


def crop_last_two_dims(x, crop_pos):
  """Cropping operation to the last two dimensions
  """
  assert(len(crop_pos) == 4 and len(x.shape) >= 2)
  return x[..., crop_pos[0] : crop_pos[1], crop_pos[2] : crop_pos[3]]


def rfft2d(x, norm='backward'):
  return rfft2(ifftshift(x), norm=norm)


def irfft2d(x):
  return fftshift(irfft2(x))


def fft2d(x, norm='backward'):
  return fft2(ifftshift(x), norm=norm)


def ifft2d(x):
  return fftshift(ifft2(x))


def wiener_filter(x: torch.Tensor, K: float, from_psf: bool = False):
  if from_psf:
    psf_pad = x
    H = rfft2d(psf_pad, norm='ortho')
  else:
    H = x
  return H.conj() / (H.abs().pow(2) + K)


def calc_crop_pos(gt_shape, pad_shape):
  crop_top = (pad_shape[0] - gt_shape[0]) // 2
  crop_bottom = crop_top + gt_shape[0]
  crop_left = (pad_shape[1] - gt_shape[1]) // 2
  crop_right = crop_left + gt_shape[1]
  crop_pos = (crop_top, crop_bottom, crop_left, crop_right)
  return crop_pos


class LenslessBasicOps():
  def __init__(self, pad_shape, gt_shape):
    assert(len(pad_shape) == 2 and len(gt_shape) == 2)
    self.pad_shape = pad_shape
    self.crop_pos = calc_crop_pos(gt_shape, pad_shape)

  def pad(self, x):
    """Padding operation for measurement and gt data (which are 4D tensors)"""
    return pad_last_two_dims(x, self.pad_shape, mode='replicate')

  def rpad(self, x):
    """Replicate-padding for 4D tensor"""
    return pad_last_two_dims(x, self.pad_shape, mode='replicate')

  def zpad(self, x):
    """Zero-padding for 4D tensor"""
    return pad_last_two_dims(x, self.pad_shape, mode='constant')

  def crop(self, x):
    return crop_last_two_dims(x, self.crop_pos)

