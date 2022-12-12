import glob
import matplotlib.pyplot as plt
import torch
import torchvision
import threading
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import PIL
import random
import copy
import logging
import math
from numpy import zeros, uint8, asarray
import os
from typing import List


# functions to help simulate the no chromatic aberration correction fault
def cartesian_to_polar(data: np.ndarray) -> np.ndarray:
    width = data.shape[1]
    height = data.shape[0]
    assert (width > 2)
    assert (height > 2)
    assert (width % 2 == 1)
    assert (height % 2 == 1)
    perimeter = 2 * (width + height - 2)
    half_di_ag = math.ceil(((width ** 2 + height ** 2) ** 0.5) / 2)
    half_w = width // 2
    half_h = height // 2
    ret = np.zeros((half_di_ag, perimeter))

    ret[0:(half_w + 1), half_h] = data[half_h, half_w::-1]
    ret[0:(half_w + 1), height + width - 2 + half_h] = data[half_h, half_w:(half_w * 2 + 1)]
    ret[0:(half_h + 1), height - 1 + half_w] = data[half_h:(half_h * 2 + 1), half_w]
    ret[0:(half_h + 1), perimeter - half_w] = data[half_h::-1, half_w]

    for i in range(0, half_h):
        slope = (half_h - i) / half_w
        di_agx = ((half_di_ag ** 2) / (slope ** 2 + 1)) ** 0.5
        unit_x_step = di_agx / (half_di_ag - 1)
        unit_y_step = di_agx * slope / (half_di_ag - 1)
        for row in range(half_di_ag):
            y_step = round(row * unit_y_step)
            x_step = round(row * unit_x_step)
            if (half_h >= y_step) and half_w >= x_step:
                ret[row, i] = data[half_h - y_step, half_w - x_step]
                ret[row, height - 1 - i] = data[half_h + y_step, half_w - x_step]
                ret[row, height + width - 2 +
                    i] = data[half_h + y_step, half_w + x_step]
                ret[row, height + width + height - 3 -
                    i] = data[half_h - y_step, half_w + x_step]
            else:
                break

    for j in range(1, half_w):
        slope = half_h / (half_w - j)
        di_agx = ((half_di_ag ** 2) / (slope ** 2 + 1)) ** 0.5
        unit_x_step = di_agx / (half_di_ag - 1)
        unit_y_step = di_agx * slope / (half_di_ag - 1)
        for row in range(half_di_ag):
            y_step = round(row * unit_y_step)
            x_step = round(row * unit_x_step)
            if half_w >= x_step and half_h >= y_step:
                ret[row, height - 1 + j] = data[half_h + y_step, half_w - x_step]
                ret[row, height + width - 2 -
                    j] = data[half_h + y_step, half_w + x_step]
                ret[row, height + width + height - 3 +
                    j] = data[half_h - y_step, half_w + x_step]
                ret[row, perimeter - j] = data[half_h - y_step, half_w - x_step]
            else:
                break
    return ret


def polar_to_cartesian(data: np.ndarray, width: int, height: int) -> np.ndarray:
    assert (width > 2)
    assert (height > 2)
    assert (width % 2 == 1)
    assert (height % 2 == 1)
    perimeter = 2 * (width + height - 2)
    half_di_ag = math.ceil(((width ** 2 + height ** 2) ** 0.5) / 2)
    half_w = width // 2
    half_h = height // 2
    ret = np.zeros((height, width))

    ret[half_h, half_w::-1] = data[0:(half_w + 1), half_h]
    ret[half_h, half_w:(half_w * 2 + 1)] = data[0:(half_w + 1), height + width - 2 + half_h]
    ret[half_h:(half_h * 2 + 1), half_w] = data[0:(half_h + 1), height - 1 + half_w]
    ret[half_h::-1, half_w] = data[0:(half_h + 1), perimeter - half_w]

    for i in range(0, half_h):
        slope = (half_h - i) / half_w
        di_ag_x = ((half_di_ag ** 2) / (slope ** 2 + 1)) ** 0.5
        unit_x_step = di_ag_x / (half_di_ag - 1)
        unit_y_step = di_ag_x * slope / (half_di_ag - 1)
        for row in range(half_di_ag):
            y_step = round(row * unit_y_step)
            x_step = round(row * unit_x_step)
            if (half_h >= y_step) and half_w >= x_step:
                ret[half_h - y_step, half_w - x_step] = \
                    data[row, i]
                ret[half_h + y_step, half_w - x_step] = \
                    data[row, height - 1 - i]
                ret[half_h + y_step, half_w + x_step] = \
                    data[row, height + width - 2 + i]
                ret[half_h - y_step, half_w + x_step] = \
                    data[row, height + width + height - 3 - i]
            else:
                break

    for j in range(1, half_w):
        slope = half_h / (half_w - j)
        di_ag_x = ((half_di_ag ** 2) / (slope ** 2 + 1)) ** 0.5
        unit_x_step = di_ag_x / (half_di_ag - 1)
        unit_y_step = di_ag_x * slope / (half_di_ag - 1)
        for row in range(half_di_ag):
            y_step = round(row * unit_y_step)
            x_step = round(row * unit_x_step)
            if half_w >= x_step and half_h >= y_step:
                ret[half_h + y_step, half_w - x_step] = \
                    data[row, height - 1 + j]
                ret[half_h + y_step, half_w + x_step] = \
                    data[row, height + width - 2 - j]
                ret[half_h - y_step, half_w + x_step] = \
                    data[row, height + width + height - 3 + j]
                ret[half_h - y_step, half_w - x_step] = \
                    data[row, perimeter - j]
            else:
                break

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if ret[i, j] == 0:
                ret[i, j] = (ret[i - 1, j] + ret[i + 1, j]) / 2
    return ret


def get_gauss(n: int) -> List[float]:
    sigma = 0.3 * (n / 2 - 1) + 0.8
    r = range(-int(n / 2), int(n / 2) + 1)
    new_sum = sum([1 / (sigma * math.sqrt(2 * math.pi)) *
                   math.exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r])
    return [(1 / (sigma * math.sqrt(2 * math.pi)) *
             math.exp(-float(x) ** 2 / (2 * sigma ** 2))) / new_sum for x in r]


def vertical_gaussian(data: np.ndarray, n: int) -> np.ndarray:
    padding = n - 1
    width = data.shape[1]
    height = data.shape[0]
    padded_data = np.zeros((height + padding * 2, width))
    padded_data[padding: -padding, :] = data
    ret = np.zeros((height, width))
    kernel = None
    old_radius = - 1
    for i in range(height):
        radius = round(i * padding / (height - 1)) + 1
        # Recreate new kernel only if we have to
        if radius != old_radius:
            old_radius = radius
            kernel = np.tile(get_gauss(1 + 2 * (radius - 1)),
                             (width, 1)).transpose()
        ret[i, :] = np.sum(np.multiply(
            padded_data[padding + i - radius + 1:padding + i + radius, :], kernel), axis=0)
    return ret


def add_chromatic(im, strength: float = 1, no_blur: bool = False):
    r, g, b = im.split()
    rdata = np.asarray(r)
    gdata = np.asarray(g)
    bdata = np.asarray(b)
    if no_blur:
        # channels remain unchanged
        r_final = r
        g_final = g
        b_final = b
    else:
        r_polar = cartesian_to_polar(rdata)
        g_polar = cartesian_to_polar(gdata)
        b_polar = cartesian_to_polar(bdata)
        blur_amount = (im.size[0] + im.size[1] - 2) / 100 * strength
        if round(blur_amount) > 0:
            r_polar = vertical_gaussian(r_polar, round(blur_amount))
            g_polar = vertical_gaussian(g_polar, round(blur_amount * 1.2))
            b_polar = vertical_gaussian(b_polar, round(blur_amount * 1.4))
        r_cartes = polar_to_cartesian(
            r_polar, width=rdata.shape[1], height=rdata.shape[0])
        g_cartes = polar_to_cartesian(
            g_polar, width=gdata.shape[1], height=gdata.shape[0])
        b_cartes = polar_to_cartesian(
            b_polar, width=bdata.shape[1], height=bdata.shape[0])

        r_final = Image.fromarray(np.uint8(r_cartes), 'L')
        g_final = Image.fromarray(np.uint8(g_cartes), 'L')
        b_final = Image.fromarray(np.uint8(b_cartes), 'L')
    g_final = g_final.resize((round((1 + 0.018 * strength) * rdata.shape[1]),
                              round((1 + 0.018 * strength) * rdata.shape[0])), Image.ANTIALIAS)
    b_final = b_final.resize((round((1 + 0.044 * strength) * rdata.shape[1]),
                              round((1 + 0.044 * strength) * rdata.shape[0])), Image.ANTIALIAS)

    r_width, r_height = r_final.size
    g_width, g_height = g_final.size
    b_width, b_height = b_final.size
    rh_diff = (b_height - r_height) // 2
    rw_diff = (b_width - r_width) // 2
    gh_diff = (b_height - g_height) // 2
    gw_diff = (b_width - g_width) // 2

    # Centre the channels
    im = Image.merge("RGB", (
        r_final.crop((-rw_diff, -rh_diff, b_width - rw_diff, b_height - rh_diff)),
        g_final.crop((-gw_diff, -gh_diff, b_width - gw_diff, b_height - gh_diff)),
        b_final))

    # Crop the image to the original image dimensions
    return im.crop((rw_diff, rh_diff, r_width + rw_diff, r_height + rh_diff))


def add_jitter(im, pixels: int = 1):
    if pixels == 0:
        return im.copy()
    r, g, b = im.split()
    r_width, r_height = r.size
    g_width, g_height = g.size
    b_width, b_height = b.size
    im = Image.merge("RGB", (
        r.crop((pixels, 0, r_width + pixels, r_height)),
        g.crop((0, 0, g_width, g_height)),
        b.crop((-pixels, 0, b_width - pixels, b_height))))
    return im


def blend_images(im, og_im, alpha: float = 1, strength: float = 1):
    og_im.putalpha(int(255 * alpha))
    og_im = og_im.resize((round((1 + 0.018 * strength) * og_im.size[0]),
                          round((1 + 0.018 * strength) * og_im.size[1])), Image.ANTIALIAS)

    h_diff = (og_im.size[1] - im.size[1]) // 2
    wdiff = (og_im.size[0] - im.size[0]) // 2
    og_im = og_im.crop((wdiff, h_diff, wdiff + im.size[0], h_diff + im.size[1]))
    im = im.convert('RGBA')

    final_im = Image.new("RGBA", im.size)
    final_im = Image.alpha_composite(final_im, im)
    final_im = Image.alpha_composite(final_im, og_im)
    final_im = final_im.convert('RGB')
    return final_im


# the fault injection class where the function is implemented
class FaultsInjectionSystem:
    error_activation_dict = {1: 'banding',
                             2: 'condensation',
                             3: 'crack',
                             4: 'dirt',
                             5: 'fog',
                             6: 'ice',
                             7: 'rain',
                             8: 'black',
                             9: 'Blur',
                             10: 'Brightness',
                             11: 'No Chromatic Aberration Correction',
                             12: 'Dead pixels',
                             13: 'Dead pixels line',
                             14: 'No Bayer filter',
                             15: 'No Demosaicing',
                             16: 'Speckle noise',
                             17: 'Sharpness',
                             18: 'darkness'
                             }
    banding = False
    condensation = False
    crack = False
    dirt = False
    fog = False

    ice = False
    rain = False
    black = False  # done
    blur = False
    brightness = False

    no_chrom_ac = False  # no_chromatic_aberration_correction
    dead_pixels = False
    dead_pixels_line = False
    no_bayer_filter = False

    no_Demosaicing = False
    speckle_noise = False
    sharpness = False
    darkness = False
    faults_activation_array = [False, False, False, False, False,
                               False, False, False, False, False,
                               False, False, False, False,
                               False, False, False, False]

    banding_iter = 0
    condensation_iter = 0
    crack_iter = 0
    dirt_iter = 0
    fog_iter = 0
    ice_iter = 0
    rain_iter = 0
    black_iter = 0
    blur_iter = 0
    brightness_iter = 0
    no_chrom_ac_iter = 0
    dead_pixels_iter = 0
    dead_pixels_line_iter = 0
    no_bayer_filter_iter = 0
    no_Demosaicing_iter = 0
    speckle_noise_iter = 0
    sharpness_iter = 0
    darkness_iter = 0

    def iter_null(self):
        self.banding_iter = 0
        self.condensation_iter = 0
        self.crack_iter = 0
        self.dirt_iter = 0
        self.fog_iter = 0
        self.ice_iter = 0
        self.rain_iter = 0
        self.black_iter = 0
        self.blur_iter = 0
        self.brightness_iter = 0
        self.no_chrom_ac_iter = 0
        self.dead_pixels_iter = 0
        self.dead_pixels_line_iter = 0
        self.no_bayer_filter_iter = 0
        self.no_Demosaicing_iter = 0
        self.speckle_noise_iter = 0
        self.sharpness_iter = 0
        self.darkness_iter = 0

    def update_fault_activation_array(self):
        self.faults_activation_array = [self.banding, self.condensation, self.crack, self.dirt, self.fog,
                                        self.ice, self.rain, self.black, self.blur, self.brightness,
                                        self.no_chrom_ac, self.dead_pixels, self.dead_pixels_line, self.no_bayer_filter,
                                        self.no_Demosaicing, self.speckle_noise, self.sharpness, self.darkness]

    def generate_fault_message(self):
        print('Generating data of faults: ')
        i = 0
        for fault_id in range(len(self.faults_activation_array)):
            if self.faults_activation_array[fault_id]:
                i = i + 1
                print(str(i) + '-', self.error_activation_dict[fault_id + 1])
            elif fault_id == len(self.faults_activation_array) - 1:
                if i == 0:
                    print("No faults selected and nothing to generate")

    # different faults depending on overlaying transparent image to the original frame like fog rain and dust etc.
    def overlay_faults(self, original_sample, error_type, strength=1):
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            current_directory = os.getcwd()
            overlay_fault_directory = current_directory + '/overlay_faults/'
            try:
                overlay_fault_directory = overlay_fault_directory + error_type + "/" + error_type + "_" + str(
                    strength) + ".png"
            except Exception as error_log:
                logging.exception(error_log)
                overlay_fault_directory = overlay_fault_directory + error_type + "/" + error_type + "_" + str(
                    strength) + ".jpg"
            effect_sample = Image.open(overlay_fault_directory).convert(original_sample.mode)
            effect_sample = effect_sample.resize(original_sample.size)
            effect_sample = Image.fromarray(cv2.cvtColor(np.asarray(effect_sample), cv2.COLOR_BGR2RGB))
            faulty_sample = Image.blend(original_sample, effect_sample, 0.35)
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(faulty_sample), cv2.COLOR_BGR2RGB))

        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ", error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # generating black image
    def black_fault(self, original_sample, error_type='black'):
        faulty_sample = original_sample.copy()
        try:
            pixels = faulty_sample.load()
            width, height = original_sample.size
            for i in range(0, width):
                for j in range(0, height):
                    pixels[i, j] = (0, 0, 0)
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ",
                  error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # generating blured image
    def blur_fault(self, original_sample, error_type="Blur", strength=1):
        weight = 2
        faulty_sample = original_sample
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            faulty_sample = cv2.blur(np.asarray(original_sample), (strength * weight, strength * weight))
            faulty_sample = Image.fromarray(cv2.cvtColor(faulty_sample, cv2.COLOR_BGR2RGB))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ", error_type.capitalize() + " fault not applied")
        return faulty_sample

    # generating brightness fault images
    def brightness_fault(self, original_sample, error_type="Brightness", strength=1):
        weight = 1
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            original_sample = Image.fromarray(cv2.cvtColor(np.asarray(original_sample), cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Brightness(original_sample)
            # enhancer = Image.fromarray(cv2.cvtColor(np.asarray(enhancer), cv2.COLOR_BGR2RGB))
            faulty_sample = enhancer.enhance(weight * strength)
            faulty_sample = Image.fromarray(np.asarray(faulty_sample))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ",
                  error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate the no chromatic aberration correction fault
    def chromatic_aberration_correction_fault(self, original_sample,
                                              error_type='No Chromatic Aberration Correction',
                                              strength=1):
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            if original_sample.size[0] % 2 == 0 or original_sample.size[1] % 2 == 0:
                if original_sample.size[0] % 2 == 0:
                    original_sample = original_sample.crop((0, 0, original_sample.size[0] - 1, original_sample.size[1]))
                    original_sample.load()
                if original_sample.size[1] % 2 == 0:
                    original_sample = original_sample.crop((0, 0, original_sample.size[0], original_sample.size[1] - 1))
                    original_sample.load()
            faulty_sample = add_chromatic(original_sample, strength * 4, no_blur=True)
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(faulty_sample), cv2.COLOR_BGR2RGB))

        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ",
                  error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate dead pixels fault
    def dead_pixels_fault(self, original_sample, error_type='Dead pixels', strength=1):
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            faulty_sample = np.array(original_sample.copy())
            h, w, _ = faulty_sample.shape
            h1 = 2  # int((h-24) / 40)  # get first height point and set w1 as height shift value
            # countpixel = 0
            for y in range(0, strength + 17):
                w1 = 2  # w1 stored value for original w2
                h1 = h1 + (2 * strength)  # spacing in height
                for x in range(0, 7 + (strength * 5)):
                    faulty_sample[h1, w1] = (0, 0, 0)  # RGB
                    # countpixel = countpixel + 1
                    w1 = w1 + strength + 2  # spacing in width
            # print(countpixel)
            # faulty_sample = Image.fromarray(faulty_sample)
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(faulty_sample), cv2.COLOR_BGR2RGB))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ",
                  error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate a line of dead pixels fault
    def dead_pixels_line_fault(self, original_sample, error_type='Dead pixels line', strength=1):
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            w = 30
            h = 30
            faulty_sample = np.array(original_sample.copy())
            for x in range(0, strength * strength * strength + 50):
                faulty_sample[h + x, w + x] = (0, 0, 0)
            # faulty_sample = Image.fromarray(faulty_sample)
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(faulty_sample), cv2.COLOR_BGR2RGB))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ",
                  error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate the no Bayer filter fault
    def no_bayer_filter_fault(self, original_sample, error_type='No Bayer filter'):
        try:
            # faulty_sample = original_sample.convert('LA') give one channel will not match our AI model
            img = np.asarray(original_sample)
            gray = cv2.cvtColor(img, cv2.CAP_PVAPI_PIXELFORMAT_BGR24)
            copy_sample = np.zeros_like(img)
            copy_sample[:, :, 0] = gray
            copy_sample[:, :, 1] = gray
            copy_sample[:, :, 2] = gray
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(copy_sample), cv2.COLOR_BGR2RGB))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ",
                  error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate no demosaicing fault

    def no_demosaicing_fault(self, original_sample, error_type='No Demosaicing'):
        try:
            original_sample = np.asarray(original_sample)
            image_width, image_height, _ = original_sample.shape
            faulty_sample = zeros((2 * image_width, 2 * image_height, 3), dtype=uint8)
            faulty_sample[::2, ::2, 2] = original_sample[:, :, 2]
            faulty_sample[1::2, ::2, 1] = original_sample[:, :, 1]
            faulty_sample[::2, 1::2, 1] = original_sample[:, :, 1]
            faulty_sample[1::2, 1::2, 0] = original_sample[:, :, 0]
            faulty_sample = cv2.cvtColor(faulty_sample, cv2.COLOR_BGR2RGB)
            faulty_sample = Image.fromarray(faulty_sample, "RGB")
            faulty_sample = faulty_sample.resize((image_height, image_height), Image.ANTIALIAS)
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ", error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate speckle noise fault
    def speckle_noise_fault(self, original_sample, error_type='Speckle noise', strength=1):
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            original_sample = np.asarray(original_sample)
            gauss = np.random.normal(0, strength * 0.28, original_sample.size)
            gauss = gauss.reshape(original_sample.shape[0], original_sample.shape[1],
                                  original_sample.shape[2]).astype('uint8')
            faulty_sample = original_sample + original_sample * gauss
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(faulty_sample), cv2.COLOR_BGR2RGB))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ", error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate sharpness fault
    def sharpness_fault(self, original_sample, error_type='Sharpness', strength=1):
        try:
            if strength > 5:
                strength = 5
            elif strength < 1:
                strength = 1
            else:
                strength = strength
            weight = 5
            enhancer = ImageEnhance.Sharpness(original_sample)
            faulty_sample = enhancer.enhance(weight * strength)
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(faulty_sample), cv2.COLOR_BGR2RGB))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred ",
                  error_type.capitalize() + " fault not applied")
            faulty_sample = original_sample
        return faulty_sample

    # simulate darkness fault
    def darkness_fault(self, original_sample, error_type="Darkness", strength=1):
        try:
            if strength < 1:
                strength = 1
            elif strength > 5:
                strength = 5
            else:
                strength = strength
            weight = (1/strength) - 0.08
            enhancer = ImageEnhance.Brightness(original_sample)
            faulty_sample = enhancer.enhance(weight)
            faulty_sample = Image.fromarray(cv2.cvtColor(np.asarray(faulty_sample), cv2.COLOR_BGR2RGB))
        except Exception as error_log:
            logging.exception(error_log)
            print("Error occurred fault not applied")
            faulty_sample = original_sample
        return faulty_sample

        # frame editing without overlaying
        # black no Bayer filter and no Demosaicing is one strength effect

    def save_image_to_disk(self, image, error_type, strength, output_path, iter):
        path_folder = output_path + error_type + '/'
        try:
            if strength is None:
                image_name = str(iter) + '_' + error_type + '.jpg'
                cv2.imwrite(os.path.join(path_folder, image_name), asarray(image))
            else:
                image_name = str(iter) + '_' + error_type + '_st_' + str(strength) + '.jpg'
                cv2.imwrite(os.path.join(path_folder, image_name), asarray(image))
        except:
            if strength is None:
                image = Image.fromarray(image)
                image_name = str(iter) + '_' + error_type + '.jpg'
                cv2.imwrite(os.path.join(path_folder, image_name), asarray(image))
            else:
                image = Image.fromarray(image)
                image_name = str(iter) + '_' + error_type + '_st_' + str(strength) + '.jpg'
                cv2.imwrite(os.path.join(path_folder, image_name), asarray(image))

    # noinspection PyTypeChecker
    def faults_generating(self, image, strength_list=None, output_path=None):
        if strength_list is None:
            strength_list = []
        if output_path is None:
            output_path = []
        if self.black:
            # iter = 0
            fault_black = self.black_fault(image)
            self.black_iter = self.black_iter + 1
            if not os.path.exists(output_path + '/black/'):
                os.makedirs(output_path + '/black/')
            self.save_image_to_disk(fault_black, error_type='black',
                                    strength=None, output_path=output_path, iter=self.black_iter)

        if self.no_bayer_filter:
            fault_no_bayer_filter = self.no_bayer_filter_fault(image)
            self.no_bayer_filter_iter = self.no_bayer_filter_iter + 1
            if not os.path.exists(output_path + '/noBayerFilter/'):
                os.makedirs(output_path + '/noBayerFilter/')
            self.save_image_to_disk(fault_no_bayer_filter, error_type='noBayerFilter',
                                    strength=None, output_path=output_path, iter=self.no_bayer_filter_iter)

        if self.no_Demosaicing:
            fault_no_Demosaicing = self.no_demosaicing_fault(image)
            self.no_Demosaicing_iter = self.no_Demosaicing_iter + 1
            if not os.path.exists(output_path + '/noDemosaicing/'):
                os.makedirs(output_path + '/noDemosaicing/')
            self.save_image_to_disk(fault_no_Demosaicing, error_type='noDemosaicing',
                                    strength=None, output_path=output_path, iter=self.no_Demosaicing_iter)

        for strength in strength_list:
            if self.blur:
                fault_blur = self.blur_fault(image, strength=strength)
                self.blur_iter = self.blur_iter + 1
                if not os.path.exists(output_path + '/blur/'):
                    os.makedirs(output_path + '/blur/')
                self.save_image_to_disk(fault_blur, error_type='blur', strength=strength,
                                        output_path=output_path, iter=self.blur_iter)

            if self.brightness:
                fault_brightness = self.brightness_fault(image, strength=strength)
                self.brightness_iter = self.brightness_iter + 1
                if not os.path.exists(output_path + '/brightness/'):
                    os.makedirs(output_path + '/brightness/')
                self.save_image_to_disk(fault_brightness, error_type='brightness', strength=strength,
                                        output_path=output_path, iter=self.brightness_iter)

            if self.no_chrom_ac:
                fault_no_chrom_ac = self.chromatic_aberration_correction_fault(image, strength=strength)
                self.no_chrom_ac_iter = self.no_chrom_ac_iter + 1
                if not os.path.exists(output_path + '/noChromaticAberrationCorrection/'):
                    os.makedirs(output_path + '/noChromaticAberrationCorrection/')
                self.save_image_to_disk(fault_no_chrom_ac, error_type='noChromaticAberrationCorrection',
                                        strength=strength, output_path=output_path, iter=self.no_chrom_ac_iter)

            if self.dead_pixels:
                fault_dead_pixels = self.dead_pixels_fault(image, strength=strength)
                self.dead_pixels_iter = self.dead_pixels_iter + 1
                if not os.path.exists(output_path + '/deadPixels/'):
                    os.makedirs(output_path + '/deadPixels/')
                self.save_image_to_disk(fault_dead_pixels, error_type='deadPixels', strength=strength,
                                        output_path=output_path, iter=self.dead_pixels_iter)

            if self.dead_pixels_line:
                fault_dead_pixels_line = self.dead_pixels_line_fault(image, strength=strength)
                self.dead_pixels_line_iter = self.dead_pixels_line_iter + 1
                if not os.path.exists(output_path + '/deadPixelsLine/'):
                    os.makedirs(output_path + '/deadPixelsLine/')
                self.save_image_to_disk(fault_dead_pixels_line, error_type='deadPixelsLine', strength=strength,
                                        output_path=output_path, iter=self.dead_pixels_line_iter)

            if self.speckle_noise:
                fault_speckle_noise = self.speckle_noise_fault(image, strength=strength)
                self.speckle_noise_iter = self.speckle_noise_iter + 1
                if not os.path.exists(output_path + '/speckleNoise/'):
                    os.makedirs(output_path + '/speckleNoise/')
                self.save_image_to_disk(fault_speckle_noise, error_type='speckleNoise', strength=strength,
                                        output_path=output_path, iter=self.speckle_noise_iter)

            if self.sharpness:
                fault_sharpness = self.sharpness_fault(image, strength=strength)
                self.sharpness_iter = self.sharpness_iter + 1
                if not os.path.exists(output_path + '/sharpness/'):
                    os.makedirs(output_path + '/sharpness/')
                self.save_image_to_disk(fault_sharpness, error_type='sharpness', strength=strength,
                                        output_path=output_path, iter=self.sharpness_iter)
            if self.darkness:
                fault_darkness = self.darkness_fault(image, strength=strength)
                self.darkness_iter = self.darkness_iter + 1
                if not os.path.exists(output_path + '/darkness/'):
                    os.makedirs(output_path + '/darkness/')
                self.save_image_to_disk(fault_darkness, error_type='darkness', strength=strength,
                                        output_path=output_path, iter=self.darkness_iter)

            # Overlaying a transparent image effect to the frame
            if self.banding:
                fault_banding = self.overlay_faults(image, error_type='banding', strength=strength)
                self.banding_iter = self.banding_iter + 1
                if not os.path.exists(output_path + '/banding/'):
                    os.makedirs(output_path + '/banding/')
                self.save_image_to_disk(fault_banding, error_type='banding', strength=strength,
                                        output_path=output_path, iter=self.banding_iter)

            if self.condensation:
                fault_condensation = self.overlay_faults(image, error_type='condensation', strength=strength)
                self.condensation_iter = self.condensation_iter + 1
                if not os.path.exists(output_path + '/condensation/'):
                    os.makedirs(output_path + '/condensation/')
                self.save_image_to_disk(fault_condensation, error_type='condensation', strength=strength,
                                        output_path=output_path, iter=self.condensation_iter)

            if self.crack:
                fault_crack = self.overlay_faults(image, error_type='crack', strength=strength)
                self.crack_iter = self.crack_iter + 1
                if not os.path.exists(output_path + '/crack/'):
                    os.makedirs(output_path + '/crack/')
                self.save_image_to_disk(fault_crack, error_type='crack', strength=strength, output_path=output_path,
                                        iter=self.crack_iter)

            if self.dirt:
                fault_dirt = self.overlay_faults(image, error_type='dirt', strength=strength)
                self.dirt_iter = self.dirt_iter + 1
                if not os.path.exists(output_path + '/dirt/'):
                    os.makedirs(output_path + '/dirt/')
                self.save_image_to_disk(fault_dirt, error_type='dirt', strength=strength, output_path=output_path,
                                        iter=self.dirt_iter)

            if self.fog:
                fault_fog = self.overlay_faults(image, error_type='fog', strength=strength)
                self.fog_iter = self.fog_iter + 1
                if not os.path.exists(output_path + '/fog/'):
                    os.makedirs(output_path + '/fog/')
                self.save_image_to_disk(fault_fog, error_type='fog', strength=strength, output_path=output_path,
                                        iter=self.fog_iter)

            if self.ice:
                fault_ice = self.overlay_faults(image, error_type='ice', strength=strength)
                self.ice_iter = self.ice_iter + 1
                if not os.path.exists(output_path + '/ice/'):
                    os.makedirs(output_path + '/ice/')
                self.save_image_to_disk(fault_ice, error_type='ice', strength=strength, output_path=output_path,
                                        iter=self.ice_iter)

            if self.rain:
                fault_rain = self.overlay_faults(image, error_type='rain', strength=strength)
                self.rain_iter = self.rain_iter + 1
                if not os.path.exists(output_path + '/rain/'):
                    os.makedirs(output_path + '/rain/')
                self.save_image_to_disk(fault_rain, error_type='rain', strength=strength, output_path=output_path,
                                        iter=self.rain_iter)
