import numpy as np

def _as_float(img):
  img = np.asarray(img, dtype=np.float32)
  if img.max() > 1.: img /= 255.
  return img


def _check_shape(img1: np.ndarray, img2: np.ndarray):
  if img1.shape != img2.shape:
    raise ValueError(f"Shape mismatch: {img1.shape} vs. {img2.shape}")


def rmse(img1: np.ndarray, img2: np.ndarray):
  img1, img2 = _as_float(img1), _as_float(img2)
  _check_shape(img1, img2)
  return np.sqrt(np.mean((img1 - img2) ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray, max_val=None):
  img1, img2 = _as_float(img1), _as_float(img2)
  _check_shape(img1, img2)
  if max_val is None:
    max_val = 255. if np.asarray(img1).max() > 1. else 1.
  mse_val = np.mean((img1 - img2) ** 2)
  if mse_val == 0: return float('inf')
  return 20 * np.log10(max_val) - 10 * np.log10(mse_val)


def _gaussian_1d(win_size=11, sigma=1.5):
  g = np.arange(win_size, dtype=np.float64)
  g -= (win_size - 1) / 2.0
  g = np.exp(-g ** 2 / (2 * sigma ** 2))
  return g / g.sum()


def _ssim_single_channel(x, y, win, k1=0.01, k2=0.03, L=1.0):
  mu1 = _convolve_gaussian(x, win)
  mu2 = _convolve_gaussian(y, win)

  mu1_sq  = mu1 * mu1
  mu2_sq  = mu2 * mu2
  mu1_mu2 = mu1 * mu2

  sigma1_sq = _convolve_gaussian(x * x, win) - mu1_sq
  sigma2_sq = _convolve_gaussian(y * y, win) - mu2_sq
  sigma12   = _convolve_gaussian(x * y, win) - mu1_mu2

  c1 = (k1 * L) ** 2
  c2 = (k2 * L) ** 2

  numerator   = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
  denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
  ssim_map = numerator / denominator
  return np.mean(ssim_map)

def _convolve_gaussian(img, win):
  out = np.apply_along_axis(lambda m: np.convolve(m, win, mode='valid'), axis=1, arr=img)
  out = np.apply_along_axis(lambda m: np.convolve(m, win, mode='valid'), axis=0, arr=out)
  return out

def ssim(img1: np.ndarray, img2: np.ndarray, win_size=11, sigma=1.5):
  img1, img2 = _as_float(img1), _as_float(img2)
  _check_shape(img1, img2)

  win = _gaussian_1d(win_size, sigma)
  if img1.ndim == 2:
    return _ssim_single_channel(img1, img2, win)
  elif img1.ndim == 3:
    channels = img1.shape[-1]
    if channels != 3:
      raise ValueError("Channel have to be the last dimension")
    return np.mean([
      _ssim_single_channel(img1[..., c], img2[..., c], win)
      for c in range(channels)
    ])
  else:
    raise ValueError("Unsupported image ndim: {}".format(img1.ndim))

