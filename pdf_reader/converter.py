from typing import *
import os
from subprocess import call

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar, LTFigure, LTPage, Rect
from pdfminer.pdfpage import PDFPage
import pytesseract

from pdf_reader.custom_dataclasses import PdfReaderConfig
from pdf_reader.pdf_page import ParseePdfPage


def decrypt_pdf_with_qpdf(path):
    # make temp file path
    repaired_file_path = path[0:-4] + "_repaired.pdf"

    # convert
    call('qpdf --password=%s --decrypt %s %s' % ('', path, repaired_file_path), shell=True)

    # delete old file
    os.remove(path)

    # rename repaired file
    os.rename(repaired_file_path, path)


def get_pdf_pages(pdf_path: str, config: Optional[PdfReaderConfig] = None) -> List[ParseePdfPage]:
    config = PdfReaderConfig(None, None, None) if config is None else config
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

    pages = []
    for page_index, page in enumerate(PDFPage.create_pages(document)):
        interpreter.process_page(page)
        layout = device.get_result()
        text_boxes = parse_layout(layout)
        if needs_ocr(text_boxes):
            pass # TODO
        else:
            page_obj = ParseePdfPage(page_index, pdf_path, page.mediabox, text_boxes, config)

        pages.append(page_obj)

    fp.close()

    return pages


def parse_layout(layout_obj: any) -> List[LTTextBox]:
    all_relevant_elements = []
    """Function to recursively parse the layout tree and flatten LTFigure elements."""
    for lt_obj in layout_obj:
        if isinstance(lt_obj, LTFigure):
            all_relevant_elements += parse_layout(lt_obj)  # Recursive
        elif isinstance(lt_obj, LTTextBox):
            all_relevant_elements.append(lt_obj)
    return all_relevant_elements


# page needs OCR if either no elements found or unreadable characters are present
def needs_ocr(text_boxes: List[LTTextBox]) -> bool:

    if len(text_boxes) == 0:
        return True

    for element in text_boxes:
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


def repair_layout():
    pass # TODO
    """

    if self.repair_layout_active:
        # first: check if page contains unreadable characters -> launch page repair if so


    temp_folder = tempfile.TemporaryDirectory()
    temp_folder_path = temp_folder.name

    # save temporary image of PDF
    target_height = 3200
    img_path_tmp = make_images_from_pdf(self.path, temp_folder_path, [target_height], self.page_index)[target_height][0]

    # start tesseract
    pdf_path_tmp = os.path.join(temp_folder_path, "ocr.pdf")
    f = open(pdf_path_tmp, "w+b")
    pdf = pytesseract.image_to_pdf_or_hocr(img_path_tmp, extension='pdf')
    f.write(bytearray(pdf))
    f.close()

    # extract chars from converted doc
    extractor_tmp_reader = PdfReader(pdf_path_tmp)
    extractor_tmp = extractor_tmp_reader.pages[0]
    extractor_tmp.extract_chars_only(create_space_chars=True)

    # create char map with "translations"
    char_map = {}
    for element in self.layout_elements:
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
                            if isinstance(c, LTChar) and t.startswith("(cid:"):
                                # find best overlapping repaired char
                                x0 = c.x0 * self.scale_multiplier
                                x1 = c.x1 * self.scale_multiplier
                                y0 = c.y0 * self.scale_multiplier
                                y1 = c.y1 * self.scale_multiplier

                                if c.get_text() not in char_map:
                                    char_map[c.get_text()] = {"taken": None, "candidates": {}}
                                char_detected, overlap = extractor_tmp.find_char(x0, x1, y0, y1)
                                if char_detected is None:
                                    continue
                                if char_detected.text not in char_map[c.get_text()]["candidates"]:
                                    char_map[c.get_text()]["candidates"][char_detected.text] = {"count": 0,
                                                                                                "score": 0}
                                char_map[c.get_text()]["candidates"][char_detected.text]['count'] += 1
                                char_map[c.get_text()]["candidates"][char_detected.text]['score'] += overlap

    # determine best entries from char map
    for key, translation_dict in char_map.items():
        if len(translation_dict['candidates'].items()) == 0:
            continue
        if len(translation_dict['candidates'].items()) == 1:
            translation_dict["taken"] = list(translation_dict['candidates'].items())[0][0]
        else:
            max_occurrences = max([x[1]['count'] for x in list(translation_dict['candidates'].items())])
            candidates_filtered = list(
                filter(lambda x: x[1]['count'] == max_occurrences, list(translation_dict['candidates'].items())))
            if len(candidates_filtered) == 1:
                translation_dict["taken"] = candidates_filtered[0][0]
            else:
                # take max score
                max_score = max([x[1]['score'] for x in candidates_filtered])
                candidates_final = list(filter(lambda x: x[1]['score'] == max_score, candidates_filtered))
                translation_dict["taken"] = candidates_final[0][0]

    # delete temporary folder and contents
    shutil.rmtree(temp_folder_path)

    # now replace from translation dict
    default_char_not_found = "X"
    for element in self.layout_elements:
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
                            if isinstance(c, LTChar) and t.startswith("(cid:"):
                                if t in char_map:
                                    c._text = char_map[t]['taken']
                                else:
                                    c._text = default_char_not_found
                                    
    """




