#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2
import numpy as np
import pywt
import pickle
from typing import Tuple, Union



# In[18]:


# ===============================================================
# watermark_models.py equivalent (VISIBLE + INVISIBLE)
# ===============================================================
import cv2
import numpy as np
import pywt
import pickle
import types

# Define module-like namespace for consistent pickling
watermark_module = types.ModuleType("watermark_models")


# In[19]:


# ===============================================================
# INVISIBLE WATERMARK MODEL
# ===============================================================
class InvisibleWatermarker:
    def __init__(self, alpha=0.05, wavelet='haar'):
        self.alpha = alpha
        self.wavelet = wavelet

    def apply(self, main_img, watermark):
        main_img = cv2.resize(main_img, (512, 512))
        watermark = cv2.resize(watermark, (128, 128))
        main_img = np.float32(main_img)
        watermark = np.float32(watermark)

        watermarked_channels = []
        for c in range(3):
            LL, (LH, HL, HH) = pywt.dwt2(main_img[:, :, c], self.wavelet)
            wm_resized = cv2.resize(watermark[:, :, c], LL.shape)
            LL_wm = LL + self.alpha * wm_resized
            watermarked_channels.append(pywt.idwt2((LL_wm, (LH, HL, HH)), self.wavelet))

        return np.uint8(np.clip(cv2.merge(watermarked_channels), 0, 255))


# In[20]:


# ===============================================================
# VISIBLE WATERMARK MODEL
# ===============================================================
class VisibleWatermarker:
    def __init__(self, alpha=0.4, wavelet='haar'):
        self.alpha = alpha
        self.wavelet = wavelet

    def apply(self, main_img, watermark):
        main_img = cv2.resize(main_img, (512, 512))
        watermark = cv2.resize(watermark, (128, 128))
        main_img = np.float32(main_img)
        watermark = np.float32(watermark)

        visible_channels = []
        for c in range(3):
            LL, (LH, HL, HH) = pywt.dwt2(main_img[:, :, c], self.wavelet)
            wm_resized = cv2.resize(watermark[:, :, c], LL.shape)
            LL_visible = cv2.addWeighted(LL, 1.0, wm_resized, self.alpha, 0.0)
            visible_channels.append(pywt.idwt2((LL_visible, (LH, HL, HH)), self.wavelet))

        return np.uint8(np.clip(cv2.merge(visible_channels), 0, 255))


# In[21]:


# ===============================================================
# IMAGE PREPROCESSOR MODEL
# ===============================================================
class ImagePreprocessor:
    def __init__(self, level=1, to_y=True, target_size=None):
        self.level = level
        self.to_y = to_y
        self.target_size = target_size

    def _read_image(self, path_or_array: Union[str, np.ndarray], force_gray: bool = False) -> np.ndarray:
        if isinstance(path_or_array, np.ndarray):
            img = path_or_array.copy()
            if force_gray and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(path_or_array, cv2.IMREAD_UNCHANGED if not force_gray else cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
        return img

    def _ensure_divisible_by_2pow(self, img: np.ndarray, level: int) -> Tuple[np.ndarray, tuple]:
        h, w = img.shape[:2]
        factor = 2 ** level
        pad_h = (factor - (h % factor)) % factor
        pad_w = (factor - (w % factor)) % factor
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)
        return padded, ((top, bottom), (left, right))

    def _convert_to_y_channel(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.ndim == 2:
            return img_bgr
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 0]

    def apply(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        orig = self._read_image(image_input)
        orig_color = orig.copy() if orig.ndim == 3 else None
        y = self._convert_to_y_channel(orig) if orig.ndim == 3 and self.to_y else orig.copy()
        if self.target_size is not None:
            y = cv2.resize(y, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        y_padded, _ = self._ensure_divisible_by_2pow(y, self.level)
        y_eq = cv2.equalizeHist(y_padded.astype(np.uint8))
        if orig_color is not None:
            y_eq = cv2.cvtColor(y_eq, cv2.COLOR_GRAY2BGR)
        return y_eq



# In[22]:


# ===============================================================
# IMAGE PREPROCESSOR MODEL
# ===============================================================
class ImagePreprocessor:
    def __init__(self, level=1, to_y=False, target_size=None):
        self.level = level
        self.to_y = to_y  # not used now, color preserved
        self.target_size = target_size

    def _read_image(self, path_or_array: Union[str, np.ndarray], force_gray: bool = False) -> np.ndarray:
        if isinstance(path_or_array, np.ndarray):
            img = path_or_array.copy()
        else:
            img = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
        return img

    def _ensure_divisible_by_2pow(self, img: np.ndarray, level: int) -> Tuple[np.ndarray, tuple]:
        h, w = img.shape[:2]
        factor = 2 ** level
        pad_h = (factor - (h % factor)) % factor
        pad_w = (factor - (w % factor)) % factor
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)
        return padded, ((top, bottom), (left, right))

    def apply(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        - Reads input
        - Applies per-channel histogram equalization
        - Pads image for DWT compatibility
        - Returns enhanced color image
        """
        orig = self._read_image(image_input)

        # Optional resize
        if self.target_size is not None:
            orig = cv2.resize(orig, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

        # Pad for DWT compatibility
        padded, _ = self._ensure_divisible_by_2pow(orig, self.level)

        # Apply per-channel histogram equalization
        eq_channels = []
        for c in cv2.split(padded):
            eq_channels.append(cv2.equalizeHist(c))
        equalized_img = cv2.merge(eq_channels)

        return np.uint8(np.clip(equalized_img, 0, 255))


# ===============================================================
# SAVE MODELS AS PICKLE FILES
# ===============================================================
with open("invisible_watermarker.pkl", "wb") as f:
    pickle.dump(InvisibleWatermarker(), f)

with open("visible_watermarker.pkl", "wb") as f:
    pickle.dump(VisibleWatermarker(), f)

with open("image_preprocessing.pkl", "wb") as f:
    pickle.dump(ImagePreprocessor(), f)

print("✅ All pickle files created successfully:")
print("   - invisible_watermarker.pkl")
print("   - visible_watermarker.pkl")
print("   - image_preprocessing.pkl")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




