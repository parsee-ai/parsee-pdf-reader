from PIL import Image, ImageDraw, ImageFont


class Displayer:

    def __init__(self, output_path, size, jpg_path=None):

        self.path = output_path
        if jpg_path is None:
            self.im = Image.new('RGB', (round(size[2]), round(size[3])), (255, 255, 255))
        else:
            self.im = Image.open(jpg_path)
        self.draw = ImageDraw.Draw(self.im, 'RGBA')
        self.page_height = size[3] - size[1]
        self.page_width = size[2] - size[0]
        self.font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 8)

    def draw_elements(self, element_list, color=None, text_list=None, outline=None):

        for k, box in enumerate(element_list):
            self.draw.rectangle([(box.x0, self.page_height - box.y1), (box.x1, self.page_height - box.y0)], fill=color,
                                outline=outline)
            if text_list is not None:
                self.draw.text((box.x0, self.page_height - box.y1), str(text_list[k]), font=self.font,
                               fill=(0, 0, 0, 1))

    def create_image(self):
        self.im.save(self.path, "PNG")
