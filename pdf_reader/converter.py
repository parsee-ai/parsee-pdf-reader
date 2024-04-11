from typing import *
import os
from subprocess import call
import tempfile
import shutil

import pypdf
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar, LTFigure, LTPage, Rect, PDFFont, PDFColorSpace, PDFGraphicState
from pdfminer.pdfpage import PDFPage
import pytesseract
from pytesseract import Output
import cv2

from pdf_reader.custom_dataclasses import PdfReaderConfig, NaturalTextHelper
from pdf_reader.pdf_page import ParseePdfPage
from pdf_reader.helper import make_images_from_pdf

"""
Currently the image detection is not very sophisticated and goes purely by file extensions
This is in order not to add some additional packages like python-magic just for this simple check
"""


def is_image(file_path: str):
    image_types_supported = ["png", "jpg", "jpeg"]
    file_name = os.path.basename(file_path).lower()
    if "." in file_name:
        ending = file_name.rsplit(".", 1)
        if ending[-1] in image_types_supported:
            return True
    return False


def decrypt_pdf_with_qpdf(path):
    # make temp file path
    repaired_file_path = path[0:-4] + "_repaired.pdf"

    # convert
    call('qpdf --password=%s --decrypt %s %s' % ('', path, repaired_file_path), shell=True)

    # delete old file
    os.remove(path)

    # rename repaired file
    os.rename(repaired_file_path, path)


def open_pdf(pdf_path: str) -> Tuple[PDFDocument, PDFPageInterpreter, PDFPageAggregator, any, pypdf.PdfReader]:
    # Open a PDF file.
    fp = open(pdf_path, 'rb')
    # Create a PDF parser object associated with the file object.
    parser = PDFParser(fp)
    # try to open document normally
    try:
        document = PDFDocument(parser, password='')
    except PDFEncryptionError:
        fp.close()
        # try to convert doc with qpdf
        decrypt_pdf_with_qpdf(pdf_path)
        # open converted file
        fp = open(pdf_path, 'rb')
        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # try to open document again
        document = PDFDocument(parser, password='')
    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()
    # Create a PDF device object.
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # pypdf
    pypdf_reader = pypdf.PdfReader(pdf_path)
    return document, interpreter, device, fp, pypdf_reader


def get_natural_text(reader: pypdf.PdfReader, page_index: int) -> NaturalTextHelper:
    try:
        pypdf_text = reader.pages[page_index].extract_text()
    except Exception:
        pypdf_text = None
    return NaturalTextHelper(pypdf_text)


def get_pdf_pages(pdf_path: str, config: Optional[PdfReaderConfig] = None, force_ocr: bool = False, **kwargs) -> List[ParseePdfPage]:
    config = PdfReaderConfig(None, None, None) if config is None else config
    # check if file is an image
    if is_image(pdf_path):
        mediabox, text_boxes = get_elements_from_image(pdf_path)
        return [ParseePdfPage(0, pdf_path, mediabox, text_boxes, config, NaturalTextHelper(None))]
    document, interpreter, device, fp, pypdf_reader = open_pdf(pdf_path)
    pages = []
    for page_index, page in enumerate(PDFPage.create_pages(document)):
        interpreter.process_page(page)
        layout = device.get_result()
        text_boxes = parse_layout(layout)
        run_ocr = force_ocr or needs_ocr(text_boxes)
        if run_ocr:
            mediabox, text_boxes, pypdf_text = repair_layout(pdf_path, page_index)
            page_obj = ParseePdfPage(page_index, pdf_path, mediabox, text_boxes, config, pypdf_text)
        else:
            page_obj = ParseePdfPage(page_index, pdf_path, page.mediabox, text_boxes, config, get_natural_text(pypdf_reader, page_index))
        pages.append(page_obj)

    fp.close()
    return pages


def parse_layout(layout_obj: any, force_chars: bool = False) -> List[Union[LTTextBox, LTChar]]:
    all_relevant_elements = []
    """Function to recursively parse the layout tree and flatten LTFigure elements."""
    for lt_obj in layout_obj:
        if isinstance(lt_obj, LTFigure):
            all_relevant_elements += parse_layout(lt_obj)  # Recursive
        elif isinstance(lt_obj, LTTextBox):
            if force_chars:
                all_relevant_elements += parse_layout(lt_obj._objs)
            else:
                all_relevant_elements.append(lt_obj)
        elif isinstance(lt_obj, LTChar):
            all_relevant_elements.append(lt_obj)
        elif isinstance(lt_obj, LTTextLine):
            all_relevant_elements += parse_layout(lt_obj._objs)
    return all_relevant_elements


# page needs OCR if either no elements found or unreadable characters are present
def needs_ocr(text_boxes: List[Union[LTTextBox, LTChar]]) -> bool:
    if len(text_boxes) == 0:
        return True

    for element in text_boxes:
        if isinstance(element, LTTextBox):
            for o in element._objs:
                if isinstance(o, LTTextLine):
                    try:
                        text = o.get_text()
                    except Exception as e:
                        text = ""
                    text_stripped = text.strip()
                    if text_stripped:
                        for kk, c in enumerate(o._objs):
                            t = c.get_text()
                            if t is not None and t.startswith("(cid:"):
                                return True
    return False


def get_elements_from_image(image_path: str) -> Tuple[Rect, List[LTChar]]:
    CONF_THRESHOLD = 60

    output = []

    # start tesseract
    img = cv2.imread(image_path)
    height = img.shape[0]
    width = img.shape[1]
    tesseract_data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 11")
    for k, conf in enumerate(tesseract_data["conf"]):
        if conf < 0:
            continue
        x0 = tesseract_data["left"][k]
        x1 = tesseract_data["left"][k] + tesseract_data["width"][k]
        y0 = tesseract_data["top"][k]
        y1 = tesseract_data["top"][k] + tesseract_data["height"][k]
        if CONF_THRESHOLD > conf >= 0:
            # crop the text and resize, then run tesseract just on that text piece again
            padding = 2
            crop_img = img[max(y0 - padding, 0):min(y1 + padding, height), max(x0 - padding, 0):min(x1 + padding, width)]
            crop_img = cv2.resize(crop_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            data_cropped_img = pytesseract.image_to_data(crop_img, output_type=Output.DICT, config="--psm 7")
            new_text = " ".join([x for k, x in enumerate(data_cropped_img["text"]) if data_cropped_img["conf"][k] >= 0])
            # replace in dictionary
            tesseract_data["text"][k] = new_text
        text = tesseract_data["text"][k]

        img_height = img.shape[0]
        el = LTChar((1, 0, 0, 1, x0, y0), PDFFont({}, {}), 1.0, 1.0, 1.0, text, 1.0, 1.0, PDFColorSpace("", 0), PDFGraphicState())
        el.x0 = x0
        el.x1 = x1
        el.y0 = img_height - y1
        el.y1 = img_height - y0

        output.append(el)

    page_size = (0, 0, img.shape[1], img.shape[0])
    return page_size, output


def repair_layout(path: str, page_index: int) -> Tuple[any, List[LTChar], NaturalTextHelper]:
    temp_folder = tempfile.TemporaryDirectory()
    temp_folder_path = temp_folder.name

    # save temporary image of PDF
    target_height = 2000
    img_path_tmp = make_images_from_pdf(path, temp_folder_path, [target_height], page_index)[target_height][0]

    mediabox, elements = get_elements_from_image(img_path_tmp)
    shutil.rmtree(temp_folder_path)
    return mediabox, elements, NaturalTextHelper(None)
