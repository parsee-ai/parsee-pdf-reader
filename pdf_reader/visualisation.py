from typing import *

from pdf_reader.custom_dataclasses import Rectangle, ExtractedPdfElement
from pdf_reader.helper import make_images_from_pdf
from pdf_reader.pdf_page import ParseePdfPage


from PIL import Image, ImageDraw, ImageFont


class Displayer:

    def __init__(self, output_path: str, size: Rectangle, jpg_path: Optional[str]):

        self.path = output_path
        if jpg_path is None:
            self.im = Image.new('RGB', (round(size.width()), round(size.height())), (255, 255, 255))
        else:
            self.im = Image.open(jpg_path)
        self.draw = ImageDraw.Draw(self.im, 'RGBA')
        self.page_height = size.height()
        self.page_width = size.width()
        self.font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 8)

    def draw_elements(self, element_list: List[Rectangle], color: Optional[Tuple]=None, outline: Optional[Tuple]=None):

        for k, box in enumerate(element_list):
            self.draw.rectangle([(box.x0, self.page_height - box.y1), (box.x1, self.page_height - box.y0)], fill=color,
                                outline=outline)

    def create_image(self):
        self.im.save(self.path, "PNG")


def visualise_elements(page: ParseePdfPage, elements: List[Rectangle], output_image_dir: str):

    color = (220, 0, 0, 50)

    # make jpg from page
    size = page.page_size.width() if page.has_wide_layout() else page.page_size.height()
    paths = make_images_from_pdf(page.pdf_path, output_image_dir, [size], page.page_index)
    jpg_path = paths[size][0]
    d = Displayer(jpg_path, page.page_size, jpg_path)
    d.draw_elements(elements, color, color)
    d.create_image()


