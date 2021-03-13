from __future__ import division

try:
    import accimage
except ImportError:
    accimage = None

import cv2
import numpy as np
from functools import partial
from PIL import Image, ImageOps, ImageEnhance
import scipy.ndimage.interpolation as itpl

class ImageAugmenter():

    def __init__(self, augmentor_configs, logger=None):
        self.cfg = augmentor_configs
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def _is_numpy_image(self, image):
        return isinstance(image, np.ndarray) and (image.ndim in {2, 3})

    def _is_pil_image(self, image):
        if accimage is not None:
            return isinstance(image, (Image.Image, accimage.Image))
        else:
            return isinstance(image, Image.Image)

    def lrflip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.lrflip, config=config)
        keys = config.KEYS
        for key in keys:
            if data_dict[key] is None: continue
            if len(data_dict[key].shape) == 3:
                data_dict[key] = data_dict[key][:, ::-1, :] - np.zeros_like(data_dict[key]) # H, W, C
            elif len(data_dict[key].shape) == 2:
                data_dict[key] = data_dict[key][:, ::-1] - np.zeros_like(data_dict[key]) # H, W, C
            else:
                raise NotImplementedError
        return data_dict

    def _crop(self, images, ctr, size, multi_scale_of=1):
        cropped_images = []
        x, y = ctr
        w, h = size
        u, v = int(x - w / 2.), int(y - h / 2.)
        if w % multi_scale_of != 0:
            w = (w // multi_scale_of) * multi_scale_of
            r = w % multi_scale_of
            u = u - r // 2
        if h % multi_scale_of != 0:
            h = (h // multi_scale_of) * multi_scale_of
            r = h % multi_scale_of
            v = v - r // 2
        for image in images:
            if len(image.shape) == 3:
                cropped_image = image[v:v+h, u:u+w, :]
            elif len(image.shape) == 2:
                cropped_image = image[v:v+h, u:u+w]
            cropped_images.append(cropped_image)
        return cropped_images

    def crop(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.crop, config=config)
        keys = config.KEYS
        ratio = config.RATIO
        assert ratio <= 1 and ratio > 0.5
        ratio = np.random.rand() * (1 - ratio) + ratio
        multi_scale_of = config.MULTI_SCALE_OF
        keep_origin_size = config.KEEP_ORIGIN_SIZE
        center_crop = config.CENTER_CROP
        h, w, _ = data_dict['image'].shape
        size = (int(w * ratio), int(h * ratio))
        if center_crop:
            ctr = (int(w / 2.), int(h / 2.))
        else:
            offset_u = int((np.random.rand() - 0.5) * (w - int(w * ratio)))
            offset_v = int((np.random.rand() - 0.5) * (h - int(h * ratio)))
            ctr = (int(w / 2.) + offset_u, int(h / 2.) + offset_v)
        for key in keys:
            if data_dict[key] is None: continue
            data_dict[key] = self._crop([data_dict[key]], ctr, size, multi_scale_of=multi_scale_of)[0]
            if keep_origin_size:
                interpolation = cv2.INTER_LINEAR if key == 'image' else cv2.INTER_NEAREST
                data_dict[key] = cv2.resize(data_dict[key], (w, h), interpolation=interpolation)
        return data_dict

    # def rotate(self, images, angle):
    #     rotated_images = []
    #     for image in images:
    #         # order=0 means nearest-neighbor type interpolation
    #         rotated_image = np.transpose(image, (1, 2, 0))
    #         rotated_image = itpl.rotate(rotated_image, angle, reshape=False, prefilter=False, order=0)
    #         rotated_image = np.transpose(rotated_image, (2, 0, 1))
    #         rotated_images.append(rotated_image)
    #     return rotated_images

    def _adjust_brightness(self, image, brightness_factor):
        if not self._is_pil_image(image):
            raise TypeError('image should be PIL Image. Got {}'.format(type(image)))

        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        return image

    def _adjust_contrast(self, image, contrast_factor):
        if not self._is_pil_image(image):
            raise TypeError('image should be PIL Image. Got {}'.format(type(image)))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        return image

    def _adjust_saturation(self, image, saturation_factor):
        if not self._is_pil_image(image):
            raise TypeError('image should be PIL Image. Got {}'.format(type(image)))

        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        return image

    def _adjust_hue(self, image, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

        if not self._is_pil_image(image):
            raise TypeError('image should be PIL Image. Got {}'.format(type(image)))

        input_mode = image.mode
        if input_mode in {'L', '1', 'I', 'F'}:
            return image

        h, s, v = image.convert('HSV').split()

        np_h = np.array(h, dtype=np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over='ignore'):
            np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, 'L')

        image = Image.merge('HSV', (h, s, v)).convert(input_mode)
        return image

    # def _adjust_gamma(self, image, gamma, gain=1):
    #     if not self._is_pil_image(image):
    #         raise TypeError('image should be PIL Image. Got {}'.format(type(image)))
    #
    #     if gamma < 0:
    #         raise ValueError('Gamma should be a non-negative real number')
    #
    #     input_mode = image.mode
    #     image = image.convert('RGB')
    #
    #     np_image = np.array(image, dtype=np.float32)
    #     np_image = 255 * gain * ((np_image / 255) ** gamma)
    #     np_image = np.uint8(np.clip(np_image, 0, 255))
    #
    #     image = Image.fromarray(np_image, 'RGB').convert(input_mode)
    #     return image

    def jit_color(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.jit_color, config=config)

        keys = config.KEYS
        brightness = config.BRIGHTNESS
        contrast = config.CONTRAST
        saturation = config.SATURATION
        hue = config.HUE

        def get_params(brightness, contrast, saturation, hue):
            transforms = []
            if brightness:
                brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
                transforms.append(lambda image: self._adjust_brightness(image, brightness_factor))
                # if self.logger is not None:
                #     self.logger.info('\tcolor jit: brightness_factor = {}'.format(brightness_factor))

            if contrast:
                contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
                transforms.append(lambda image: self._adjust_contrast(image, contrast_factor))
                # if self.logger is not None:
                #     self.logger.info('\tcolor jit: contrast_factor = {}'.format(contrast_factor))

            if saturation:
                saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
                transforms.append(lambda image: self._adjust_saturation(image, saturation_factor))
                # if self.logger is not None:
                #     self.logger.info('\tcolor jit: saturation_factor = {}'.format(saturation_factor))

            if hue:
                hue_factor = np.random.uniform(-hue, hue)
                transforms.append(lambda image: self._adjust_hue(image, hue_factor))
                # if self.logger is not None:
                #     self.logger.info('\tcolor jit: hue_factor = {}'.format(hue_factor))

            np.random.shuffle(transforms)

            return transforms

        transforms = get_params(brightness, contrast, saturation, hue)

        for key in keys:
            if data_dict[key] is None: continue
            assert len(data_dict[key].shape) == 3
            if not (self._is_numpy_image(data_dict[key])):
                raise TypeError('image should be ndarray. Got {}'.format(type(data_dict[key])))

            data_dict[key] = Image.fromarray(data_dict[key].astype(np.uint8))
            for t in transforms:
                data_dict[key] = t(data_dict[key])

            data_dict[key] = np.array(data_dict[key])
        return data_dict

    def forward(self, data_dict):
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict