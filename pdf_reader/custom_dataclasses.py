import copy
import json
from typing import *
from dataclasses import dataclass

from pdf_reader.helper import clean_numeric_value


class PdfReaderConfig:

    SPACE_MAX_DISTANCE = 6
    TOLERANCE_GEN = 10
    LINE_BREAK_MAX_DISTANCE = 14

    # default page size, all coordinates will be transformed to fit these default max widths
    page_default_width_normal = 594
    page_default_width_horizontal = 1200

    tolerance_top_bot = 3
    tolerance_top_bot_line_items = 6
    char_dist_max = 1
    line_break_distance = 20
    separate_columns_distance = 150
    separate_columns_distance_wide_layout = 250
    separate_table_distance = 50
    text_max_number_col = 15
    table_header_max_height = 120
    table_sides_tolerance_x0 = 50
    table_sides_tolerance_x1 = 50
    tolerance_columns_li = 20
    line_max_height = 2

    def __init__(self, space_max_distance: Union[int, None], tolerance_gen: Union[int, None], line_break_distance: Union[int, None]):
        if space_max_distance is not None:
            self.SPACE_MAX_DISTANCE = space_max_distance

        if tolerance_gen is not None:
            self.TOLERANCE_GEN = tolerance_gen

        if line_break_distance is not None:
            self.LINE_BREAK_MAX_DISTANCE = line_break_distance


@dataclass
class RelativeAreaPrediction:
    class_name: str
    x0: float
    x1: float
    y0: float
    y1: float
    prob: float


class Rectangle:

    def __init__(self, x0: Union[int, None], x1: Union[int, None], y0: Union[int, None], y1: Union[int, None]):

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        self.elements = []
        self.x0_el = None
        self.x1_el = None
        self.y0_el = None
        self.y1_el = None

        self.in_group = False

        self.tolerance_detection = 3

    def __str__(self):
        return "R: (x0: " + str(self.x0) + ", x1: " + str(self.x1) + ", y0: " + str(self.y0) + ", y1: " + str(
            self.y1) + ")"

    def __repr__(self):
        el_str = ", ".join(["(" + str(x) + ")" for x in self.elements])
        return "R: (x0: " + str(self.x0) + ", x1: " + str(self.x1) + ", y0: " + str(self.y0) + ", y1: " + str(
            self.y1) + ", elements: [" + el_str + "])"

    def list(self):
        return [round(self.x0), round(self.x1), round(self.y0), round(self.y1)]

    def dict_coordinates(self):
        return {"x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1}

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0

    def x_distance_to(self, element):

        if self.h_overlap(element, "", 0):
            return 0

        if element.x0 >= self.x1:
            return element.x0 - self.x1
        else:
            return abs(self.x0 - element.x1)

    def y_distance_to(self, element):

        if self.v_overlap(element, "", 0):
            return 0

        if element.y0 >= self.y1:
            return element.y0 - self.y1
        else:
            return abs(self.y0 - element.y1)

    def v_overlap(self, element, field_add="", custom_tolerance=None):

        tolerance = custom_tolerance if custom_tolerance is not None else self.tolerance_detection

        # make element a bit smaller for detection
        y0_chosen = getattr(self, "y0" + field_add) + tolerance
        y1_chosen = getattr(self, "y1" + field_add) - tolerance

        v_overlaps = True

        if (y1_chosen < element.y0) or (y0_chosen > element.y1):
            v_overlaps = False
        return v_overlaps

    def list_index(self, area_list):
        for a, area in enumerate(area_list):
            if area.x0 == self.x0 and area.x1 == self.x1 and area.y0 == self.y0 and area.y1 == self.y1:
                return a
        return -1

    def h_inside(self, element, field_add="", custom_tolerance=None):

        tolerance = custom_tolerance if custom_tolerance is not None else self.tolerance_detection

        # make element a bit larger for detection
        x0_chosen = getattr(self, "x0" + field_add) - tolerance
        x1_chosen = getattr(self, "x1" + field_add) + tolerance

        el_x0_chosen = getattr(element, "x0" + field_add)
        el_x1_chosen = getattr(element, "x1" + field_add)

        if el_x0_chosen >= x0_chosen and el_x1_chosen <= x1_chosen:
            return True
        else:
            return False

    def h_overlap(self, element, field_add="", custom_tolerance=None):

        tolerance = custom_tolerance if custom_tolerance is not None else self.tolerance_detection

        # make element a bit smaller for detection
        x0_chosen = getattr(self, "x0" + field_add) + tolerance
        x1_chosen = getattr(self, "x1" + field_add) - tolerance

        h_overlaps = True

        if (x0_chosen > element.x1) or (x1_chosen < element.x0):
            h_overlaps = False
        return h_overlaps

    def overlap_percent(self, area):

        h_percent = self.h_overlap_percent([area])[0]
        v_percent = self.v_overlap_percent([area])[0]

        return (h_percent + v_percent) / 2

    def h_overlap_percent(self, area_list, field_add=""):

        x0_chosen = getattr(self, "x0" + field_add)
        x1_chosen = getattr(self, "x1" + field_add)

        overlaps = []
        for area in area_list:
            if not self.h_overlap(area, field_add, 0):
                overlaps.append(0)
                continue
            if x0_chosen <= area.x0 and x1_chosen >= area.x1:
                overlaps.append(area.width() / self.width())
                continue
            if area.x0 <= x0_chosen and area.x1 >= x1_chosen:
                overlaps.append(1)
                continue
            if x0_chosen <= area.x0:
                overlaps.append(min(1, (x1_chosen - area.x0) / self.width()))
            else:
                # x0_chosen > area.x0
                overlaps.append(min(1, (area.x1 - x0_chosen) / self.width()))

        return overlaps

    def v_overlap_percent(self, area_list, field_add=""):

        y0_chosen = getattr(self, "y0" + field_add)
        y1_chosen = getattr(self, "y1" + field_add)

        overlaps = []
        for area in area_list:
            if not self.v_overlap(area, field_add, 0):
                overlaps.append(0)
                continue
            if y0_chosen <= area.y0 and y1_chosen >= area.y1:
                overlaps.append(area.height() / self.height())
                continue
            if area.y0 <= y0_chosen and area.y1 >= y1_chosen:
                overlaps.append(1)
                continue
            if y0_chosen <= area.y0:
                overlaps.append(min(1, (y1_chosen - area.y0) / self.height()))
            else:
                # y0_chosen > area.y0
                overlaps.append(min(1, (area.y1 - y0_chosen) / self.height()))

        return overlaps

    def collides_with(self, element, field_add=""):

        return self.h_overlap(element, field_add) and self.v_overlap(element, field_add)

    def is_inside(self, element, custom_tolerance=None):

        tolerance = custom_tolerance if custom_tolerance is not None else self.tolerance_detection

        # make element a bit larger for detection
        x0_chosen = self.x0 - tolerance
        x1_chosen = self.x1 + tolerance
        y0_chosen = self.y0 - tolerance
        y1_chosen = self.y1 + tolerance

        if element.x0 >= x0_chosen and element.x1 <= x1_chosen and element.y0 >= y0_chosen and element.y1 <= y1_chosen:
            return True
        else:
            return False


class AreaPrediction(Rectangle):

    def __init__(self, relative_area: RelativeAreaPrediction, page_width: int, page_height: int, class_id: int):
        super().__init__(int(relative_area.x0*page_width), int(relative_area.x1*page_width), int((1-relative_area.y1)*page_height), int((1-relative_area.y0)*page_height))
        self.class_value = relative_area.class_name
        self.class_id = class_id
        self.prob = relative_area.prob


class ExtractedPdfElement(Rectangle):

    def __init__(self, x0: int, x1: int, y0: int, y1: int, dict_el: Dict):
        super().__init__(x0, x1, y0, y1)
        self.dict_el: Dict = dict_el

    def __str__(self):
        return str(self.dict_el)

    def __repr__(self):
        return self.__str__()

    def get_text(self) -> str:
        return self.dict_el["t"] if "t" in self.dict_el else ""


class ExtractedFigure(ExtractedPdfElement):

    def __init__(self, x0: int, x1: int, y0: int, y1: int):
        super().__init__(x0, x1, y0, y1, {"type": "ef"})

    def __str__(self):
        return str(self.dict_el)

    def __repr__(self):
        return self.__str__()

    def get_text(self) -> str:
        return "[figure]"


class PdfParagraph(ExtractedPdfElement):

    def __init__(self, elements: List[ExtractedPdfElement], config: PdfReaderConfig):
        super().__init__(0, 0, 0, 0, {"type": "em"})
        self.config = config
        self.text = ""
        self.line_break_char = "\n"
        self.elements: List[ExtractedPdfElement] = elements
        self.fit_area_from_elements()
        self.add_text_from_elements()
        self.area_group_ids = []

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)

    def get_text(self) -> str:
        return self.text

    def to_dict(self):
        return {"text": self.text, "coordinates": self.dict_coordinates()}

    def toJSON(self):
        return json.dumps(self.to_dict())

    def add_text_from_elements(self):
        self.text = ""
        elements_sorted = sorted(self.elements, key=lambda el: (-el.y1, el.x0))
        for k, el in enumerate(elements_sorted):
            if k > 0 and self.elements[k - 1].y1 - el.y1 > self.config.LINE_BREAK_MAX_DISTANCE:
                self.text += self.line_break_char
            elif k > 0:
                self.text += " "
            self.text += el.dict_el["t"]

    def fit_area_from_elements(self):
        area = make_area_from_elements(self.elements)
        self.x0 = area.x0
        self.x1 = area.x1
        self.y0 = area.y0
        self.y1 = area.y1
        self.area_group_ids = set([el.dict_el["in_area"].class_id for el in self.elements if "in_area" in el.dict_el and el.dict_el["in_area"] is not None])

    def add_el(self, el: ExtractedPdfElement):
        self.elements.append(el)
        self.fit_area_from_elements()
        self.add_text_from_elements()


class SingleChar(Rectangle):

    def __init__(self, x0, x1, y0, y1, text, scale_multiplier=1):
        super().__init__(x0 * scale_multiplier, x1 * scale_multiplier, y0 * scale_multiplier, y1 * scale_multiplier)

        self.text = text

    def __str__(self):
        return str(self.text) + " [" + str(round(self.x0, 2)) + "," + str(round(self.x1, 2)) + "," + str(
            round(self.y0, 2)) + "," + str(round(self.y1, 2)) + "]"

    def __repr__(self):
        return str(self)


def make_area_from_elements(elements: List[Rectangle], field_add: str = "") -> Rectangle:
    min_x0 = None
    min_y0 = None
    max_x1 = None
    max_y1 = None

    for el in elements:
        elx0 = getattr(el, "x0" + field_add)
        elx1 = getattr(el, "x1" + field_add)
        ely0 = getattr(el, "y0" + field_add)
        ely1 = getattr(el, "y1" + field_add)

        min_x0 = elx0 if min_x0 is None or elx0 < min_x0 else min_x0
        min_y0 = ely0 if min_y0 is None or ely0 < min_y0 else min_y0
        max_x1 = elx1 if max_x1 is None or elx1 > max_x1 else max_x1
        max_y1 = ely1 if max_y1 is None or ely1 > max_y1 else max_y1

    return Rectangle(min_x0, max_x1, min_y0, max_y1)


class TableGroup(Rectangle):

    def __init__(self, elements, field_add=""):

        super().__init__(None, None, None, None)

        self.elements += elements
        self.fit_elements(field_add)

        self.data_area = None
        self.line_items = []

    def __str__(self):
        el_str = ", ".join(["(" + str(x) + ")" for x in self.elements])
        return "G: (x0: " + str(self.x0) + ", x1: " + str(self.x1) + ", y0: " + str(self.y0) + ", y1: " + str(
            self.y1) + ", elements: [" + el_str + "])"

    def __repr__(self):
        el_str = ", ".join(["(" + str(x) + ")" for x in self.elements])
        return "G: (x0: " + str(self.x0) + ", x1: " + str(self.x1) + ", y0: " + str(self.y0) + ", y1: " + str(
            self.y1) + ", elements: [" + el_str + "])"

    def put_element(self, element, field_add=""):

        element.in_group = True
        self.elements.append(element)
        self.fit_elements(field_add)

    def remove_element(self, element_or_index, field_add=""):

        if isinstance(element_or_index, int):
            self.elements[element_or_index].in_group = False
            del (self.elements[element_or_index])
        if isinstance(element_or_index, ElementMiner):
            element_or_index.in_group = False
            found = None
            for a in range(0, len(self.elements)):
                if self.elements[a].is_identical(element_or_index):
                    found = a
                    break
            if found is not None:
                del (self.elements[found])
        self.fit_elements(field_add)

    def fit_elements(self, field_add=""):
        new_area = make_area_from_elements(self.elements, field_add)
        self.x0 = new_area.x0
        self.x1 = new_area.x1
        self.y0 = new_area.y0
        self.y1 = new_area.y1


class Area(Rectangle):

    def __init__(self, x0=0, x1=0, y0=0, y1=0, tolerance_detection=3):

        super().__init__(x0, x1, y0, y1)

        self.in_group = False

        self.tolerance_detection = tolerance_detection

    def __str__(self):
        el_str = ", ".join(["(" + str(x) + ")" for x in self.elements])
        return "A: (x0: " + str(self.x0) + ", x1: " + str(self.x1) + ", y0: " + str(self.y0) + ", y1: " + str(
            self.y1) + ", elements: [" + el_str + "])"

    def __repr__(self):
        el_str = ", ".join(["(" + str(x) + ")" for x in self.elements])
        return "A: (x0: " + str(self.x0) + ", x1: " + str(self.x1) + ", y0: " + str(self.y0) + ", y1: " + str(
            self.y1) + ", elements: [" + el_str + "])"

    def break_at_horizontal(self, break_point, discard=None):

        if break_point >= self.y1:
            return self, None
        if break_point <= self.y0:
            return self, None

        former_y0 = self.y0

        if discard == "top":
            self.y1 = break_point
            return self, None
        elif discard == "bot":
            self.y0 = break_point
            return self, None
        else:
            self.y0 = break_point
            new_area = Area(self.x0, self.x1, former_y0, break_point)
            # split elements
            to_del = []
            for k, el in enumerate(self.elements):
                if not self.is_inside(el):
                    new_area.elements.append(el)
                    to_del.append(k)
            for k in range(len(self.elements) - 1, -1, -1):
                if k in to_del:
                    del (self.elements[k])

            # fit width of areas again
            self.refit_x_only()
            new_area.refit_x_only()

            return self, new_area

    def put_element(self, el):

        self.elements.append(el)
        self.fit_elements(el)

    def init_with_elements(self, el_list):

        self.elements = [x for x in el_list if x is not None]

        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None

        for el in self.elements:
            if self.x0 is None or el.x0 < self.x0:
                self.x0 = el.x0
            if self.y0 is None or el.y0 < self.y0:
                self.y0 = el.y0
            if self.x1 is None or el.x1 > self.x1:
                self.x1 = el.x1
            if self.y1 is None or el.y1 > self.y1:
                self.y1 = el.y1

    def fit_elements(self, el_fit=None):

        if el_fit is None:
            el_list = self.elements
        else:
            el_list = [el_fit]

        for el in el_list:
            if self.x0_el is None or el.x0 < self.x0_el:
                self.x0_el = el.x0
            if self.y0_el is None or el.y0 < self.y0_el:
                self.y0_el = el.y0
            if self.x1_el is None or el.x1 > self.x1_el:
                self.x1_el = el.x1
            if self.y1_el is None or el.y1 > self.y1_el:
                self.y1_el = el.y1

    def refit_x_only(self):

        if len(self.elements) == 0:
            return

        self.x0 = None
        self.x1 = None
        self.x0_el = None
        self.x1_el = None

        for el in self.elements:
            if self.x0_el is None or el.x0 < self.x0_el:
                self.x0_el = el.x0
                self.x0 = el.x0
            if self.x1_el is None or el.x1 > self.x1_el:
                self.x1_el = el.x1
                self.x1 = el.x1

    def stackable(self, el, field_add=""):

        if self.h_inside(el, field_add) or el.h_inside(self, field_add) or self.h_overlap_percent([el], field_add)[
            0] > 0.6:
            # check that elements don't collide
            joined_areas_a = self.get_joined_areas()
            joined_areas_b = el.get_joined_areas()

            for ar in joined_areas_a:
                for ar2 in joined_areas_b:
                    if ar.collides_with(ar2):
                        return False
            return True
        else:
            return False

    def get_joined_areas(self):

        els_sorted = sorted(self.elements, key=lambda x: -x.y1)

        separate_areas = []
        el_list_temp = []
        for k in range(0, len(els_sorted)):
            if k == 0:
                el_list_temp.append(els_sorted[k])
            else:
                if els_sorted[k - 1].y0 - els_sorted[k].y1 < self.tolerance_detection:
                    el_list_temp.append(els_sorted[k])
                else:
                    # create new area with previous elements
                    new_area = Area(0, 0, 0, 0)
                    new_area.init_with_elements(el_list_temp)
                    separate_areas.append(new_area)
                    el_list_temp = [els_sorted[k]]
            if k == len(els_sorted) - 1 and len(el_list_temp) > 0:
                # create new area with previous elements
                new_area = Area(0, 0, 0, 0)
                new_area.init_with_elements(el_list_temp)
                separate_areas.append(new_area)
        return separate_areas

    def combine(self, el):

        self_copy = copy.deepcopy(self)

        if el.x0 < self.x0:
            self_copy.x0 = el.x0
        if el.y0 < self.y0:
            self_copy.y0 = el.y0
        if el.x0_el is not None and self.x0_el is not None:
            if el.x0_el < self.x0_el:
                self_copy.x0_el = el.x0_el
        if el.y0_el is not None and self.y0_el is not None:
            if el.y0_el < self.y0_el:
                self_copy.y0_el = el.y0_el

        if el.x1 > self.x1:
            self_copy.x1 = el.x1
        if el.y1 > self.y1:
            self_copy.y1 = el.y1
        if el.x1_el is not None and self.x1_el is not None:
            if el.x1_el > self.x1_el:
                self_copy.x1_el = el.x1_el
        if el.y1_el is not None and self.y1_el is not None:
            if el.y1_el > self.y1_el:
                self_copy.y1_el = el.y1_el

        for el_contained in el.elements:
            self_copy.elements.append(el_contained)

        self_copy.fit_elements()

        return self_copy


class ElementMiner(Rectangle):

    def __init__(self, x0: int = 0, x1: int = 0, y0: int = 0, y1: int = 0, text: str = "", row_index: Optional[int] = None, has_bold: bool = False,
                 scale_multiplier: float = 1):

        super().__init__(int(x0 * scale_multiplier), int(x1 * scale_multiplier), int(y0 * scale_multiplier), int(y1 * scale_multiplier))

        self.text = text
        self.has_bold = has_bold

        self.row_index = row_index
        self.bounding_text = None

        self.tolerance_detection = 1

        self.in_group = False
        self.linked_el = []

    def __str__(self):
        b_val = "True" if self.has_bold else "False"
        return "t:'" + str(self.text) + "', x0: " + str(round(self.x0)) + ", x1: " + str(
            round(self.x1)) + ", y0: " + str(round(self.y0)) + ", y1: " + str(round(self.y1)) + ", b: " + b_val + ""

    def __repr__(self):
        return str(self.text) + " (" + str(round(self.x0)) + "," + str(round(self.x1)) + "," + str(
            round(self.y0)) + "," + str(round(self.y1)) + ")"

    # convert data to dict so that it can be saved as e.g. JSON
    def to_dict(self):
        return {"type": "em", "a": self.list(), "t": self.text}

    def to_extracted_element(self) -> ExtractedPdfElement:
        return ExtractedPdfElement(self.x0, self.x1, self.y0, self.y1, self.to_dict())

    def is_identical(self, element):

        if round(element.x0) == round(self.x0) and round(element.x1) == round(self.x1) and round(element.y0) == round(
                self.y0) and round(element.y1) == round(self.y1) and self.text == element.text:
            return True
        else:
            return False

    def in_list(self, el_list):
        for el in el_list:
            if self.is_identical(el):
                return True
        return False

    def collides_with(self, element, field_add=""):

        # make element a bit smaller for detection
        x0_chosen = self.x0 + self.tolerance_detection
        x1_chosen = self.x1 - self.tolerance_detection
        y0_chosen = self.y0 + self.tolerance_detection
        y1_chosen = self.y1 - self.tolerance_detection

        hoverlaps = True
        voverlaps = True
        if (x0_chosen > element.x1) or (x1_chosen < element.x0):
            hoverlaps = False
        if (y1_chosen < element.y0) or (y0_chosen > element.y1):
            voverlaps = False

        return hoverlaps and voverlaps

    def merge(self, element):

        # change text
        # check y position difference
        y_diff = abs(element.y1 - self.y1)
        if y_diff <= 2:
            # same y position
            if self.x0 < element.x0:
                self.text = self.text + " " + element.text
            else:
                self.text = element.text + " " + self.text
        else:
            if element.y1 > self.y1:
                self.text = element.text + " " + self.text
            else:
                self.text = self.text + " " + element.text

        # change coordinates
        self.x0 = element.x0 if element.x0 < self.x0 else self.x0
        self.x1 = element.x1 if element.x1 > self.x1 else self.x1
        self.y0 = element.y0 if element.y0 < self.y0 else self.y0
        self.y1 = element.y1 if element.y1 > self.y1 else self.y1

        self.row_index = min(self.row_index, element.row_index)

        self.elements += element.elements


class LineItem:

    def __init__(self, el):
        self.el = copy.deepcopy(el)
        self.el_list = [el]
        self.internal_table_index = 0
        self.caption = el.text if el is not None else ""
        self.values = []
        self.value_types = []
        self.area = Area(0, 0, 0, 0)
        if el is not None:
            self.area.init_with_elements([self.el])
        self.is_valid = None
        self.is_separator = None

    def __str__(self):
        return "LI (" + str(self.internal_table_index) + "): " + str(self.caption) + "; values: " + str(self.values)

    def __repr__(self):
        return self.__str__()

    def has_line_item_values(self):
        return False in [x.is_empty() for x in self.values]

    # adds element to line item
    def add_el(self, el):
        # merge main el
        self.el.merge(el)
        self.el_list.append(el)
        # update caption
        self.caption = self.el.text if self.el is not None else ""
        # update area
        self.area.init_with_elements(self.el_list)

    def assign_values(self, value_el_list, type_el_list):
        for k, v_el in enumerate(value_el_list):
            self.values.append(ValueItem(v_el))
            self.value_types.append(type_el_list[k])

        # determine validity and separator status

        # valid = has caption & at least one numeric value
        # if no caption, element is not valid
        if self.el is None:
            self.is_valid = False
        if self.is_valid is None:
            # checks if numeric value is present
            for k, item_type in enumerate(self.value_types):
                if item_type is not None and "num-value" in item_type:
                    self.is_valid = True
                    break
        if self.is_valid is None:
            self.is_valid = False

        # whether the line item has values that make it act as separator for other line items
        for k, item_type in enumerate(self.value_types):
            if item_type is not None and "time-year" in item_type:
                self.is_separator = True
                break
        if self.is_separator is None:
            self.is_separator = False

    def first_non_empty_value(self):
        for v in self.values:
            if not v.is_empty():
                return v
        return None

    def dict(self, simple_values=False):
        return {
            "c": self.caption,
            "a": self.area.list(),
            "v": [v.dict(simple_values) for v in self.values]
        }


class ValueItem:

    def __init__(self, el):
        self.el = el
        self.val = el.text if el is not None else ""
        self.val_clean = None
        self.make_final_value()
        self.currency = None
        self.unit = None

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return self.__str__()

    def is_empty(self):
        if self.val == "":
            return True
        return False

    def make_final_value(self):

        if self.val == "":
            return

        self.val_clean = clean_numeric_value(self.val)

    def remove_meta(self):

        self.currency = None
        self.unit = None

    def dict(self, simple=False):

        if simple:
            return {"v": self.val_clean}

        currency_valid = True
        if self.unit is not None and (self.unit.output_val() == "p" or self.unit.output_val() == "u"):
            currency_valid = False

        return {
            "v": self.val_clean,
            "c": self.currency.output_val() if self.currency is not None and self.val_clean is not None and currency_valid else None,
            "u": self.unit.output_val() if self.unit is not None and self.val_clean is not None else None
        }


class MetaRow:

    def __init__(self, el, c_overlap):
        self.elements = []
        self.area = Area(0, 0, 0, 0)
        self.row_index = el.row_index
        self.values = [None for _ in c_overlap]

        self.add_el(el, c_overlap)

    def __str__(self):
        return "META: " + str(self.values)

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return {
            "a": self.area.list(),
        }

    def values_dict(self):
        return [x.to_dict() if x is not None else {"type": "null"} for x in self.values]

    def add_el(self, el, c_overlap):

        self.elements.append(el)

        for idx, overlap in enumerate(c_overlap):
            if overlap > 0.1:
                if self.values[idx] is None:
                    self.values[idx] = el
                else:
                    # if value in cell is already occupied, merge them
                    self.values[idx].merge(el)

        # update area
        self.area.init_with_elements(self.elements)


class ExtractedTable(ExtractedPdfElement):

    table_area: Area
    total_value_area: Area

    def __init__(self, table_dict):

        self.g_index = table_dict['g_index']
        self.items = []
        # meta info above columns
        self.meta = []
        self.meta_area = None
        # all base elements that collide with or inside the table but neither part of the line items or the column meta
        self.other_contained_elements = []
        self.num_rows = 0
        self.num_cols = 0
        self.li_area = table_dict['li_area']
        self.value_areas = table_dict['value_areas']
        for li in table_dict['values']:
            self.items.append(li)
        self.set_table_size()
        self.set_table_area()

        super().__init__(self.table_area.x0, self.table_area.x1, self.table_area.y0, self.table_area.y1, self.to_dict())

    def __str__(self):
        return "T: (r: " + str(self.num_rows) + ", c:" + str(self.num_cols) + ")"

    def __repr__(self):
        return self.__str__()

    def dict_v2(self):

        return {
            "i": [li.dict() for li in self.items],
            "a": self.table_area.list(),
            "va": [v.list() for v in self.value_areas],
        }

    # convert data to dict so that it can be saved as e.g. JSON
    def to_dict(self):
        return {
            "type": "et",
            "i": [li.dict(simple_values=True) for li in self.items],
            "a": self.table_area.list(),
            "va": [v.list() for v in self.value_areas],
            "tva": self.total_value_area.list(),
            "m": [m.values_dict() for m in self.meta]
        }

    def df_format(self):

        table_data = []

        if len(table_data) > 0:
            table_data.append({"item": ""})

        for li in self.items:
            row_dict = {"item": li.caption}
            for k, col in enumerate(li.values):
                row_dict['col_' + str(k)] = col.val

            table_data.append(row_dict)

        return table_data

    def set_table_size(self):

        self.num_cols = 0
        self.num_rows = len(self.items)
        if self.num_rows > 0:
            self.num_cols = len(self.items[0].values)

    def set_table_area(self):

        # set size of table area excluding header
        table_x0 = self.li_area.x0
        table_y0 = self.li_area.y0
        table_y1 = self.li_area.y1
        table_x1 = self.value_areas[-1].x1

        self.total_value_area = Area(self.value_areas[0].x0, self.value_areas[-1].x1,
                                     min([x.y0 for x in self.value_areas]), max([x.y1 for x in self.value_areas]))

        # adjust for meta dimension if available
        if len(self.meta) > 0:
            meta_y0 = min([m.area.y0 for m in self.meta])
            meta_y1 = max([m.area.y1 for m in self.meta])
            meta_x0 = min([m.area.x0 for m in self.meta])
            meta_x1 = max([m.area.x1 for m in self.meta])

            self.meta_area = Area(meta_x0, meta_x1, meta_y0, meta_y1)

            table_y1 = max(meta_y1, table_y1)
            table_x0 = min(meta_x0, table_x0)
            table_x1 = max(meta_x1, table_x1)

        self.table_area = Area(table_x0, table_x1, table_y0, table_y1)

    def add_meta_element(self, element, c_overlap):

        if element.row_index not in [x.row_index for x in self.meta]:
            self.meta.append(MetaRow(element, c_overlap))
            self.meta = list(sorted(self.meta, key=lambda x: x.row_index))
        else:
            idx = [x.row_index for x in self.meta].index(element.row_index)
            self.meta[idx].add_el(element, c_overlap)


@dataclass
class ExtractedPage:
    index: int
    size: Rectangle
    elements: List[ExtractedPdfElement]
    paragraphs: List[ExtractedPdfElement]