import os
import re
import shutil
import tempfile
from decimal import Decimal
from subprocess import call
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from typing import List, Dict, Optional

IMG_BATCH_SIZE = 10

to_filter_numbers = re.compile(r'(\([^0-9 ]*\))|[^0-9A-Za-z/]')


def letter_len(string):
    string = re.sub('[^A-Za-z]', '', string)
    return len(string)


def words_contained(cell_str, lower=False):
    if lower:
        cell_str = cell_str.lower()
    return list(filter(lambda x: x != "", re.sub('[^A-Za-z0-9%$€£¥]', ' ', cell_str).split(" ")))


def is_number_cell(cell_str):
    if cell_str is None:
        return False
    cell_str = to_filter_numbers.sub('', cell_str)
    if cell_str.isdigit():
        return True
    else:
        return False


def scale_image(path, target_max_width_or_height, target_width=None):
    image = cv2.imread(path)
    if target_width is not None:
        resized = cv2.resize(image, (round(target_width), round(target_max_width_or_height)),
                             interpolation=cv2.INTER_AREA)
    else:
        if image.shape[0] > image.shape[1]:
            ratio = float(image.shape[0]) / float(image.shape[1])
            target_width = int(target_max_width_or_height / ratio)
            resized = cv2.resize(image, (target_width, target_max_width_or_height), interpolation=cv2.INTER_AREA)
        else:
            ratio = float(image.shape[1]) / float(image.shape[0])
            target_height = int(target_max_width_or_height / ratio)
            resized = cv2.resize(image, (target_max_width_or_height, target_height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(path, resized)


def get_target_size(width: int, height: int, target_size: int):

    if width > height:
        target_height = int((target_size / width) * height)
        target_width = target_size
    else:
        target_height = target_size
        target_width = int((target_size / height) * width)

    return target_width, target_height


def make_images_from_pdf(path_to_pdf: str, output_path: str, target_sizes: List[int], page_index_only: Optional[int]) -> Dict[int, List[str]]:

    def convert_batch(first_page_num: int, last_page_num: int):
        pages = convert_from_path(path_to_pdf, first_page=first_page_num, last_page=last_page_num, fmt="jpg")
        for page_index, image in enumerate(pages):
            for target_size in target_sizes:
                if target_size not in image_paths:
                    image_paths[target_size] = []
                image = image.resize(get_target_size(image.width, image.height, target_size))
                image_path = os.path.join(output_path, f"{target_size}_p_{page_index + first_page_num -1}.jpg")
                image.save(image_path, 'JPEG')
                image_paths[target_size].append(image_path)

    info = pdfinfo_from_path(path_to_pdf)
    max_pages = info["Pages"]
    image_paths = {}
    if page_index_only is None:
        for page_batch in range(1, max_pages + 1, IMG_BATCH_SIZE):
            convert_batch(page_batch, min(page_batch + IMG_BATCH_SIZE - 1, max_pages))
    else:
        convert_batch(page_index_only+1, page_index_only+1)

    return image_paths


def comma_separator_thousands(cell_str):
    if re.search(r'\b[0-9]{1,3}[,][0-9]{3}\b', cell_str):
        return True
    return False


def dot_separator_thousands(cell_str):
    if re.search(r'\b[0-9]{1,3}[.][0-9]{3}\b', cell_str):
        return True
    return False


def is_negative(cell_str):
    # minus
    if re.search(r'(-|—|–|‒|―|–|−)( | |)*\d', cell_str.strip()):
        return True
    # brackets
    if re.search(r'\([\d ,.%]+(\)|\b)', cell_str.strip()):
        return True
    return False


def clean_numeric_value(cell_str):
    mult = 1
    if is_negative(cell_str):
        mult = -1

    cell_str = re.sub(r'[^0-9,.]', '', cell_str)

    # clean thousands separator
    if comma_separator_thousands(cell_str):
        cell_str = re.sub(r',', "", cell_str)
    elif dot_separator_thousands(cell_str):
        cell_str = re.sub(r'\.', "", cell_str)

    # now also replace the comma with a dot should it be used in any case
    cell_str = re.sub(r',', ".", cell_str)

    if cell_str.replace('.', '', 1).isdigit():
        return Decimal(cell_str) * mult
    else:
        return None
