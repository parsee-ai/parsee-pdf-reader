from typing import *
import math
import copy

import numpy as np
from pdfminer.layout import LTTextBox, LTTextLine, LTChar, Rect

from pdf_reader.custom_dataclasses import Rectangle, BaseElement, Area, BaseElementGroup, LineItem, ValueItem, ExtractedTable, ExtractedPdfElement, PdfReaderConfig, TableGroup, NaturalTextHelper
from pdf_reader.helper import is_number_cell, letter_len, words_contained


def filter_out_empty_columns(output_list: List[ExtractedTable], min_cols: int) -> List[ExtractedTable]:
    # filter out columns which have only None values
    for table_index in range(len(output_list) - 1, -1, -1):
        table = output_list[table_index]
        if len(table.items) > 0:
            for col_index in range(len(table.items[0].values) - 1, -1, -1):
                if len([li.values[col_index] for li in table.items if
                        li.values[col_index].is_empty()]) == len(table.items):
                    # splice column
                    table.remove_column(col_index)
        if len(table.items) == 0 or len(table.items[0].values) < min_cols:
            del (output_list[table_index])

    return output_list


def has_collision_post_alignment(list1, list2, new_pos, vertical, take_min):
    list_a = copy.deepcopy(list1)
    list_b = copy.deepcopy(list2)

    if vertical:
        if take_min:
            key = "x0"
            key_o = "x1"
        else:
            key = "x1"
            key_o = "x0"
    else:
        if take_min:
            key = "y0"
            key_o = "y1"
        else:
            key = "y1"
            key_o = "y0"

    # align object
    for el in list_a:
        prior_val = getattr(el, key)
        diff = new_pos - prior_val
        if diff != 0:
            setattr(el, key, new_pos)
            setattr(el, key_o, getattr(el, key_o) + diff)
    for el in list_b:
        prior_val = getattr(el, key)
        diff = new_pos - prior_val
        if diff != 0:
            setattr(el, key, new_pos)
            setattr(el, key_o, getattr(el, key_o) + diff)

    for el in list_a:
        for el2 in list_b:
            if not el.is_identical(el2):
                if el.collides_with(el2):
                    return True
    return False


class ParseePdfPage:

    page_index: int
    page_size: Rectangle
    config: PdfReaderConfig
    scale_multiplier: float = 1
    elements_list: List[BaseElement]
    non_text_elements: List[BaseElement]
    natural_text: NaturalTextHelper

    def __init__(self, page_index: int, pdf_path: str, page_size_pdfminer: Rect, text_boxes: List[Union[LTTextBox, LTChar]], config: PdfReaderConfig, natural_text: NaturalTextHelper):

        self.page_index = page_index
        self.pdf_path = pdf_path
        self.config = config
        self._set_page_size(page_size_pdfminer)
        self._set_elements(text_boxes)
        self.natural_text = natural_text

    def _get_page_size_multiplier(self, mediabox: Rect):

        page_width = mediabox[2] - mediabox[0]
        page_height = mediabox[3] - mediabox[1]

        if page_height > page_width:
            scale_multiplier = float(self.config.page_default_width_normal / page_width)
        else:
            scale_multiplier = float(self.config.page_default_width_horizontal / page_width)

        return scale_multiplier

    def has_wide_layout(self):
        return self.page_size.width() > self.page_size.height()

    def _set_page_size(self, page_size_pdfminer: Rect):
        self.scale_multiplier = self._get_page_size_multiplier(page_size_pdfminer)
        self.page_size = Rectangle(int(page_size_pdfminer[0] * self.scale_multiplier), int(page_size_pdfminer[2] * self.scale_multiplier), int(page_size_pdfminer[1] * self.scale_multiplier), int(page_size_pdfminer[3] * self.scale_multiplier))

    def _set_elements(self, text_boxes: List[Union[LTTextBox, LTChar]]):

        self.elements_list = []
        self.non_text_elements = []

        # now collect elements
        new_text = ""
        new_x0 = None
        new_x1 = None
        new_y0 = None
        new_y1 = None
        has_bold = False

        ref = None
        last_char_ref = None
        break_now = False
        for kk, element in enumerate(text_boxes):
            if isinstance(element, LTTextBox):
                for o in element._objs:
                    if isinstance(o, LTTextLine):
                        try:
                            text = o.get_text()
                        except Exception as e:
                            text = ""
                        text_stripped = text.strip()
                        if text_stripped:

                            # treat all space separated text fragments as separate objects, or if there is a harsh break in size/y-position
                            new_text = ""
                            new_x0 = None
                            new_x1 = None
                            new_y0 = None
                            new_y1 = None
                            has_bold = False

                            ref_y0 = None
                            ref_y1 = None
                            ref_size = None
                            break_now = False
                            for kk, c in enumerate(o._objs):

                                t = c.get_text()
                                if isinstance(c, LTChar) and t != " " and t != "\t" and t != ")":
                                    if ref_y0 is None:
                                        ref_y0 = c.y0
                                        ref_y1 = c.y1
                                        ref_size = c.size
                                    elif abs(ref_y0 - c.y0) > 1 or abs(ref_y1 - c.y1) > 1 or abs(ref_size - c.size) > 3:
                                        break_now = True

                                    if not break_now:
                                        new_x0 = c.x0 if new_x0 is None or new_x0 > c.x0 else new_x0
                                        new_x1 = c.x1 if new_x1 is None or new_x1 < c.x1 else new_x1
                                        new_y0 = c.y0 if new_y0 is None else new_y0
                                        new_y1 = c.y1 if new_y1 is None else new_y1
                                        new_text += t
                                        font_name = c.fontname.lower()
                                        if "bold" in font_name:
                                            has_bold = True

                                # start new element
                                if t == " " or t == "\t" or t == ")" or break_now:
                                    if t == ")":
                                        new_text += ")"
                                        new_x1 = c.x1
                                    if new_text != "" and new_x0 is not None and new_x1 is not None:
                                        self.elements_list.append(
                                            BaseElement(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text,
                                                        has_bold=has_bold, scale_multiplier=self.scale_multiplier))
                                    has_bold = False
                                    if not break_now:
                                        new_text = ""
                                        new_x0 = None
                                        new_x1 = None
                                        new_y0 = None
                                        new_y1 = None
                                    else:
                                        new_text = t
                                        new_x0 = c.x0
                                        new_x1 = c.x1
                                        new_y0 = c.y0
                                        new_y1 = c.y1
                                        font_name = c.fontname.lower()
                                        if "bold" in font_name:
                                            has_bold = True

                                    ref_y0 = None
                                    ref_y1 = None
                                    ref_size = None
                                    break_now = False
                                if kk == len(
                                        o._objs) - 1 and new_text != "" and new_x0 is not None and new_x1 is not None:
                                    self.elements_list.append(
                                        BaseElement(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text,
                                                    has_bold=has_bold, scale_multiplier=self.scale_multiplier))
                                    # reset variables
                                    new_text = ""
                                    new_x0 = None
                                    new_x1 = None
                                    new_y0 = None
                                    new_y1 = None

            elif isinstance(element, LTChar):
                # process char elements directly
                # treat all space separated text fragments as separate objects, or if there is a harsh break in size/y-position
                c = element
                t = c.get_text()
                if t != " " and t != "\t" and t != ")":
                    if ref is None:
                        ref = c
                    elif abs(ref.y0 - c.y0) > 1 or abs(ref.y1 - c.y1) > 1 or abs(ref.size - c.size) > 3:
                        break_now = True

                    # break if y distance too far
                    if kk > 0 and abs(c.y1 - text_boxes[kk - 1].y1) > 1:
                        break_now = True

                    # break if x distance too far
                    if kk > 0 and (abs(c.x0 - text_boxes[kk - 1].x1) > self.config.char_dist_max or (
                            last_char_ref is not None and abs(c.x0 - last_char_ref.x1) > self.config.char_dist_max)):
                        break_now = True

                    if not break_now:
                        new_x0 = c.x0 if new_x0 is None or new_x0 > c.x0 else new_x0
                        new_x1 = c.x1 if new_x1 is None or new_x1 < c.x1 else new_x1
                        new_y0 = c.y0 if new_y0 is None else new_y0
                        new_y1 = c.y1 if new_y1 is None else new_y1
                        new_text += t
                        font_name = c.fontname.lower()
                        if "bold" in font_name:
                            has_bold = True

                    last_char_ref = c

                # start new element
                if (t == " " or t == "\t" or t == ")" or break_now):
                    if t == ")":
                        new_text += ")"
                        new_x1 = c.x1
                    if new_text != "" and new_x0 is not None and new_x1 is not None:
                        self.elements_list.append(
                            BaseElement(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text, has_bold=has_bold,
                                        scale_multiplier=self.scale_multiplier))
                    has_bold = False
                    if not break_now:
                        new_text = ""
                        new_x0 = None
                        new_x1 = None
                        new_y0 = None
                        new_y1 = None
                    else:
                        new_text = t
                        new_x0 = c.x0
                        new_x1 = c.x1
                        new_y0 = c.y0
                        new_y1 = c.y1
                        font_name = c.fontname.lower()
                        if "bold" in font_name:
                            has_bold = True

                    ref = None
                    break_now = False
                if kk == len(text_boxes) - 1 and new_text != "" and new_x0 is not None and new_x1 is not None:
                    self.elements_list.append(
                        BaseElement(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text, has_bold=has_bold,
                                    scale_multiplier=self.scale_multiplier))
                    # no need to reset variables as end is reached anyway
            else:
                # check if an element still has to be added
                if new_text != "" and new_x0 is not None and new_x1 is not None:
                    self.elements_list.append(
                        BaseElement(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text, has_bold=has_bold,
                                    scale_multiplier=self.scale_multiplier))
                    # reset variables
                    new_text = ""
                    new_x0 = None
                    new_x1 = None
                    new_y0 = None
                    new_y1 = None
                self.non_text_elements.append(
                    BaseElement(x0=element.x0, x1=element.x1, y0=element.y0, y1=element.y1, text="",
                                scale_multiplier=self.scale_multiplier))

    def _clean_aligned(self, dict_el, key2_co, alignment_take_min=False, custom_tolerance=None, check_collision=True):

        tolerance_chosen = self.config.TOLERANCE_GEN if custom_tolerance is None else custom_tolerance

        vertical_alignment = True
        if key2_co.startswith("y"):
            vertical_alignment = False

        key_s = key2_co + "_min" if not alignment_take_min else key2_co + "_max"

        xk_pos = list(dict_el.keys())
        len_keys = len(xk_pos)
        to_del = []

        if len_keys <= 1:
            return dict_el

        # make distance matrix
        none_row = [None for _ in range(0, len_keys)]
        dist_matrix = np.array([none_row for _ in range(0, len_keys)])

        distances_dict = {}
        for a in range(0, len_keys):
            for b in range(a + 1, len_keys):
                dist = abs(xk_pos[a] - xk_pos[b])
                if dist <= tolerance_chosen:
                    dist_matrix[a][b] = dist
                    if dist not in distances_dict:
                        distances_dict[dist] = []
                    distances_dict[dist].append((a, b))

        distances_sorted = sorted(list(distances_dict))

        for dist in distances_sorted:
            # consolidate one after the other, updating the none matrix
            to_consolidate = distances_dict[dist]
            for cons in to_consolidate:
                if dist_matrix[cons[0]][cons[1]] is not None:

                    key_taken = xk_pos[cons[0]] if dict_el[xk_pos[cons[0]]]["c"] > dict_el[xk_pos[cons[1]]]["c"] else \
                    xk_pos[cons[1]]
                    key_rejected = xk_pos[cons[0]] if key_taken == xk_pos[cons[1]] else xk_pos[cons[1]]

                    collided = False if not check_collision else has_collision_post_alignment(
                        dict_el[key_taken]['list'], dict_el[key_rejected]['list'], key_taken, vertical_alignment,
                        alignment_take_min)

                    if not collided:
                        to_del.append(key_rejected)
                        # update matrix
                        pos_to_del = xk_pos.index(key_rejected)
                        for kk in range(0, len_keys):
                            dist_matrix[pos_to_del, kk] = None
                            dist_matrix[kk, pos_to_del] = None
                        # update dictionary
                        dict_el[key_taken]["c"] += dict_el[key_rejected]["c"]
                        new_xs = dict_el[key_taken][key_s]
                        switch = dict_el[key_rejected][key_s] < dict_el[key_taken][key_s] if not alignment_take_min else \
                        dict_el[key_rejected][key_s] > dict_el[key_taken][key_s]
                        if switch:
                            new_xs = dict_el[key_rejected][key_s]
                        dict_el[key_taken][key_s] = new_xs
                        dict_el[key_taken]['list'] += dict_el[key_rejected]['list']

        return {k: v for (k, v) in dict_el.items() if k not in to_del}

    def _get_bounding_text_area(self, group: BaseElementGroup, col_elements_exclude, handled=None) -> Tuple[BaseElementGroup, List[BaseElement]]:

        if handled is None:
            handled = [] + group.elements

        row = self.rows[group.row_index]

        for el in row['list']:
            if not el.in_list(group.elements) and not el.in_list(col_elements_exclude) and not el.in_list(handled):
                if abs(el.x0 - group.x1) <= self.config.SPACE_MAX_DISTANCE and el.x1 > group.x1:
                    group.add_element(el)
                    handled.append(el)
                    return self._get_bounding_text_area(group, col_elements_exclude, handled)
                if abs(group.x0 - el.x1) <= self.config.SPACE_MAX_DISTANCE and el.x0 < group.x0:
                    group.add_element(el)
                    handled.append(el)
                    return self._get_bounding_text_area(group, col_elements_exclude, handled)

        return group, handled

    def _find_rows(self):

        dict_bot = {}

        # search for all elements bot aligned
        for element in self.elements_list:
            y0 = math.floor(element.y0)
            y1 = math.ceil(element.y1)
            if y0 not in dict_bot:
                dict_bot[y0] = {"c": 0, "y1_max": None, "list": [], "base_elements": []}
            dict_bot[y0]["c"] += 1
            dict_bot[y0]["y1_max"] = y1 if dict_bot[y0]["y1_max"] is None or dict_bot[y0]["y1_max"] < y1 else \
            dict_bot[y0]["y1_max"]
            dict_bot[y0]['list'].append(element)

        # clean given some tolerance
        dict_bot_cleaned = self._clean_aligned(dict_bot, "y1", True)

        # make list
        rows = []
        for y0_key, dict_n in dict_bot_cleaned.items():
            dict_n['y0_min'] = y0_key
            dict_n['list'] = sorted(dict_n['list'], key=lambda k: k.x0)
            rows.append(dict_n)

        rows = sorted(rows, key=lambda k: -k['y0_min'])

        # assign row index to each element
        for k, row in enumerate(rows):
            for el in row['list']:
                el.row_index = k

        self.rows = rows

        # make base elements for elements in row (bounding text)
        for k, row in enumerate(self.rows):
            base_elements = []
            handled = []
            for el in row['list']:

                if el not in handled:
                    new_el, handled = self._get_bounding_text_area(BaseElementGroup([el]), handled)
                    base_elements.append(new_el)

            row['base_elements'] = base_elements

    def _find_columns_numeric(self, min_row_limit=1):

        dict_numbers = {}

        # search for >= min_row_limit text elements left aligned, >= min_row_limit numbers right aligned
        for element in self.elements_list:
            if is_number_cell(element.text):
                # right alignment
                x0 = math.floor(element.x0)
                x1 = math.ceil(element.x1)
                if x1 not in dict_numbers:
                    dict_numbers[x1] = {"c": 0, "x0_min": None, "list": []}
                dict_numbers[x1]["c"] += 1
                dict_numbers[x1]["x0_min"] = x0 if dict_numbers[x1]["x0_min"] is None or dict_numbers[x1][
                    "x0_min"] > x0 else dict_numbers[x1]["x0_min"]
                dict_numbers[x1]['list'].append(element)

        # clean numbers given some tolerance
        dict_numbers_cleaned = self._clean_aligned(dict_numbers, "x0", False)

        # make list
        number_cols = []
        for x1_key, dict_n in dict_numbers_cleaned.items():
            if dict_n['c'] > min_row_limit:
                dict_n['x1_max'] = x1_key
                number_cols.append(dict_n)

        number_cols = sorted(number_cols, key=lambda k: k['x0_min'])

        self.number_cols_list = number_cols

    def _add_unbroken_areas(self):

        all_col_items = []

        for col_index, col in enumerate(self.number_cols_list):
            all_col_items += col['list']

        for col_index, col in enumerate(self.number_cols_list):

            el_list = []
            # remove elements that are inside a text block
            for kk, el in enumerate(col['list']):
                bounding_text, _ = self._get_bounding_text_area(BaseElementGroup([el]), all_col_items)
                if is_number_cell(bounding_text.text) or len(bounding_text.text) <= self.config.text_max_number_col:
                    el_list.append(el)

            unbroken_areas = []
            # sort items in col by y1 position
            el_list = sorted(el_list, key=lambda x: -x.y1)
            # alignment means at least 2 elements
            alignment_area_start = None
            alignment_count = 0
            element_list = []
            prev_el = None
            for kk, el in enumerate(el_list):

                alignment_count += 1
                element_list.append(el)
                if alignment_count >= 2 and alignment_area_start is None and prev_el is not None:
                    alignment_area_start = prev_el.y1

                if alignment_area_start is not None and prev_el is not None:
                    # check if something is between this element and the one before
                    area_to_check = Area(col['x0_min'], col["x1_max"], el.y1, prev_el.y0)

                    for row in self.rows:
                        break_all = False

                        if area_to_check.y1 + self.config.TOLERANCE_GEN >= row[
                            'y0_min'] >= area_to_check.y0 - self.config.TOLERANCE_GEN or area_to_check.y1 + self.config.TOLERANCE_GEN >= row[
                            'y1_max'] >= area_to_check.y0 - self.config.TOLERANCE_GEN:
                            for el_text_area in row['base_elements']:
                                # check if base element can be used to break the area (can't be used if one of its elements is part of the current column)
                                use_base_el = False if len(
                                    [x for x in el_text_area.elements if x in el_list]) > 0 else True
                                # check if base element collides with current area
                                if use_base_el and area_to_check.collides_with(
                                        el_text_area) and not area_to_check.is_inside(el_text_area):

                                    if alignment_count > 2:
                                        ua = Area(col['x0_min'], col["x1_max"], prev_el.y0, alignment_area_start)
                                        ua.elements = element_list[:-1]
                                        unbroken_areas.append(ua)
                                    alignment_area_start = el.y1
                                    alignment_count = 1
                                    element_list = [el]
                                    break_all = True
                                    break
                        if break_all:
                            break

                    if kk == len(
                            el_list) - 1 and alignment_count >= 2 and prev_el is not None and alignment_area_start is not None:
                        ua = Area(col['x0_min'], col["x1_max"], el.y0, alignment_area_start)
                        ua.elements = element_list
                        unbroken_areas.append(ua)

                prev_el = el

            col['unbroken'] = unbroken_areas

    def _extend_area_obj(self, area_org, upwards, row_index, init=False, row_index_limit=None):

        if init:
            if row_index == 0:
                return
            if row_index == len(self.rows) - 1:
                return
            row_index = row_index - 1 if upwards else row_index + 1
        else:
            if row_index < 0:
                return
            if row_index > len(self.rows) - 1:
                return

        # check that limiting row index is not reached
        if row_index_limit is not None:
            if upwards and row_index <= row_index_limit:
                return
            if not upwards and row_index >= row_index_limit:
                return

        area = copy.deepcopy(area_org)

        if upwards:
            k_1 = "y1_max"
        else:
            k_1 = "y0_min"

        relevant_row = self.rows[row_index]

        # extend y area of area (hypothetically)
        if upwards:
            area.y1 = relevant_row[k_1]
            row_index_next = row_index - 1
        else:
            area.y0 = relevant_row[k_1]
            row_index_next = row_index + 1

        for relevant_el in relevant_row['base_elements']:
            if area.collides_with(relevant_el) and not area.is_inside(relevant_el) and letter_len(relevant_el.text) > 2:
                return

        # extend y area of area (actually)
        if upwards:
            area_org.y1 = area.y1
        else:
            area_org.y0 = area.y0

        self._extend_area_obj(area_org, upwards, row_index_next, False, row_index_limit)

    def _detect_tables(self):

        # delete columns with no unbroken areas
        cols_candidates = [x for x in self.number_cols_list if len(x['unbroken']) > 0]

        # extend unbroken areas for non-numeric elements that fit inside the area
        for kk, col in enumerate(cols_candidates):
            for area_obj_idx, area_obj in enumerate(col['unbroken']):
                # determine boundaries
                # upwards
                row_index_limit = None if area_obj_idx == 0 else max(
                    [x.row_index for x in col['unbroken'][area_obj_idx - 1].elements])
                min_row_index = min([x.row_index for x in area_obj.elements])
                self._extend_area_obj(area_obj, True, min_row_index, True, row_index_limit)

                # downwards
                row_index_limit = None if area_obj_idx == len(col['unbroken']) - 1 else min(
                    [x.row_index for x in col['unbroken'][area_obj_idx + 1].elements])
                max_row_index = max([x.row_index for x in area_obj.elements])
                self._extend_area_obj(area_obj, False, max_row_index, True, row_index_limit)

        # break unbroken columns by blank line
        for k, row in enumerate(self.rows):
            if k > 0:
                distance_to_before = (row['y1_max'] - self.rows[k - 1]["y0_min"]) * -1

                if distance_to_before >= self.config.line_break_distance:
                    break_point_start = self.rows[k - 1]["y0_min"]
                    break_point_end = row['y1_max']
                    for col in cols_candidates:
                        to_add = []
                        for area_obj in col['unbroken']:
                            area_obj, new_obj = area_obj.break_at_horizontal(break_point_start)
                            if new_obj is not None:
                                new_obj, _ = new_obj.break_at_horizontal(break_point_end, "top")
                                to_add.append(new_obj)
                        col['unbroken'] += to_add

        # collect elements inside areas
        all_elements = copy.deepcopy(self.elements_list)
        all_relevant_areas = []
        for col in cols_candidates:
            for area_obj in col['unbroken']:
                num_numbers = 0
                num_text = 0
                must_have_items = area_obj.elements
                area_obj.elements = []
                for k in range(len(all_elements) - 1, -1, -1):
                    if area_obj.is_inside(all_elements[k], 5) or all_elements[k].in_list(must_have_items):
                        area_obj.put_element(all_elements[k])
                        is_number_cell_check = is_number_cell(all_elements[k].text)
                        letter_len_check = letter_len(all_elements[k].text)
                        if is_number_cell_check:
                            num_numbers += 1
                        if not is_number_cell_check and letter_len_check > 0:
                            num_text += 1
                        del (all_elements[k])
                if len(area_obj.elements) > 0 and num_numbers > 0 and num_numbers >= num_text:
                    all_relevant_areas.append(area_obj)

        # save for use in features etc.
        self.numeric_areas = all_relevant_areas

        # all element frames that share some distance on the y-axis form a group (each Area can only be in one group)
        all_relevant_areas = sorted(all_relevant_areas, key=lambda x: -x.y1_el)
        groups = []
        temp_group = None
        for k, ar in enumerate(all_relevant_areas):
            if not ar.in_group:
                if temp_group is None:
                    temp_group = TableGroup([ar], "_el")
                for kk in range(k + 1, len(all_relevant_areas)):
                    ar2 = all_relevant_areas[kk]
                    if not ar2.in_group:
                        if ar2.v_overlap(temp_group, "_el"):
                            temp_group.put_element(ar2, "_el")
                groups.append(temp_group)
                temp_group = None

        # split groups where columns are too far away from each other
        for g in groups:
            g.elements = sorted(g.elements, key=lambda x: x.x0_el)
            break_at = None
            for k, area in enumerate(g.elements):
                if k > 0:
                    if (not self.has_wide_layout() and area.x_distance_to(
                            g.elements[k - 1]) > self.config.separate_columns_distance) or (
                            self.has_wide_layout() and area.x_distance_to(
                            g.elements[k - 1]) > self.config.separate_columns_distance_wide_layout):
                        # separate group
                        break_at = k
                        break
            if break_at is not None:
                new_group_elements = []
                for kk in range(len(g.elements) - 1, break_at - 1, -1):
                    new_group_elements.append(g.elements[kk])
                    g.remove_element(kk, "_el")
                new_group = TableGroup(new_group_elements, "_el")
                groups.append(new_group)

        # combine areas inside group that are horizontally aligned
        for g in groups:
            g.elements = sorted(g.elements, key=lambda x: x.y1_el)
            handled = []
            a = 0
            while a < len(g.elements):
                if a not in handled:
                    for b in range(0, len(g.elements)):
                        if a != b and b not in handled:
                            if g.elements[a].stackable(g.elements[b], "_el"):
                                handled.append(a)
                                handled.append(b)
                                g.elements.append(g.elements[a].combine(g.elements[b]))
                                break
                a += 1

            for index_del in sorted(handled, reverse=True):
                del g.elements[index_del]

        # TABLE AREA IS THE MAX OF ALL ALIGNED UNBROKEN AREAS INSIDE GROUP
        for g in groups:
            x0 = None
            x1 = None
            y0 = None
            y1 = None
            for area in g.elements:
                min_x0 = max(area.x0, g.x0)
                max_x1 = min(area.x1, g.x1)
                min_y0 = max(area.y0, g.y0)
                max_y1 = min(area.y1, g.y1)

                x0 = min_x0 if x0 is None or min_x0 < x0 else x0
                x1 = max_x1 if x1 is None or max_x1 > x1 else x1
                y0 = min_y0 if y0 is None or min_y0 < y0 else y0
                y1 = max_y1 if y1 is None or max_y1 > y1 else y1

            if x0 is not None:
                g.data_area = Area(x0, x1, y0, y1)

        self.groups = groups

    def _find_columns_text(self, el_list, tolerance=0):

        dict_text = {}

        for element in el_list:
            # left alignment
            x0 = math.floor(element.x0)
            x1 = math.ceil(element.x1)
            if x0 not in dict_text:
                dict_text[x0] = {"c": 0, "x1_max": None, "list": []}
            dict_text[x0]["c"] += 1
            dict_text[x0]["x1_max"] = x1 if dict_text[x0]["x1_max"] is None or dict_text[x0]["x1_max"] < x1 else \
            dict_text[x0]["x1_max"]
            dict_text[x0]['list'].append(element)

        # clean numbers given some tolerance
        dict_text_cleaned = self._clean_aligned(dict_text, "x1", True, tolerance, False)

        # make list
        text_cols = []
        for x0_key, dict_n in dict_text_cleaned.items():
            dict_n['x0_min'] = x0_key
            text_cols.append(dict_n)

        return sorted(text_cols, key=lambda k: k['x0_min'])

    def _detect_line_items(self):

        # make candidates for entire page: left aligned texts
        all_elements = []
        for row in self.rows:
            for el in row['base_elements']:
                # element should not be numeric
                if not is_number_cell(el.text):
                    all_elements.append(el)

        all_candidate_cols = self._find_columns_text(all_elements, self.config.tolerance_columns_li)

        # make areas
        for col in all_candidate_cols:
            col['area'] = Area()
            col['area'].init_with_elements(col['list'])

        # for each group: find best left aligned text -> criteria: distance (the closer the better), contains some text
        candidates_by_group = {}
        for g_index, g in enumerate(self.groups):

            # get rows that actually contain values
            value_rows = []
            for area in g.elements:
                for el in area.elements:
                    if el.row_index not in value_rows:
                        value_rows.append(el.row_index)
            value_rows = sorted(value_rows)

            candidates_by_group[g_index] = []
            # check which candidates are valid for the table group
            for candidate in all_candidate_cols:
                if candidate['area'].x0 < g.x0 and candidate['area'].v_overlap(g):
                    overlapping_elements = [x for x in candidate['list'] if x.row_index in value_rows]
                    if len(overlapping_elements) == 0:
                        continue
                    overlapping_elements_area = Area()
                    overlapping_elements_area.init_with_elements(overlapping_elements)
                    candidates_by_group[g_index].append(
                        {"scoring": {"final_score": 0}, "overlapping_elements_area": overlapping_elements_area,
                         "value_rows": value_rows})

        # criteria for scoring, min letter len of line item of 6
        scoring_weights = {"words": 5, "distance": 1, "completeness": 5, "natural_text_fits": 8}
        chosen_bordering_area = {}
        for g_index, g in enumerate(self.groups):
            # if only one candidate, take that
            if len(candidates_by_group[g_index]) == 1:
                chosen_bordering_area[g_index] = candidates_by_group[g_index][0]
            elif len(candidates_by_group[g_index]) > 1:
                # sort by distance to values
                candidates_by_group[g_index] = list(
                    sorted(candidates_by_group[g_index], key=lambda x: g.x0 - x['overlapping_elements_area'].x0))
                for k, candidate_dict in enumerate(candidates_by_group[g_index]):
                    candidate_dict['scoring']['words'] = len(
                        [x for x in candidate_dict['overlapping_elements_area'].elements if len(words_contained(x.text)) > 0]) / len(candidate_dict['value_rows'])
                    candidate_dict['scoring']['completeness'] = len(
                        [x for x in candidate_dict['overlapping_elements_area'].elements]) / len(
                        candidate_dict['value_rows'])
                    candidate_dict['scoring']['distance'] = 1 if len(candidates_by_group[g_index]) == 1 else 1 - (
                                k / (len(candidates_by_group[g_index]) - 1))
                    candidate_dict['scoring']['natural_text_fits'] = self.natural_text.is_adjacent_percent(g, candidate_dict['overlapping_elements_area'])
                    for key, weight in scoring_weights.items():
                        candidate_dict['scoring']['final_score'] += candidate_dict['scoring'][key] * weight
                scores = [x['scoring']['final_score'] for x in candidates_by_group[g_index]]
                index_taken = scores.index(max(scores))
                chosen_bordering_area[g_index] = candidates_by_group[g_index][index_taken]

            # assign line items
            line_item_helper = {}
            if g_index in chosen_bordering_area:
                for row_index, row in enumerate(self.rows):
                    if row_index in chosen_bordering_area[g_index]['value_rows'] or (
                            (g.y1 > row['y1_max'] > g.y0) or (g.y1 > row['y0_min'] > g.y0)):
                        for el in row['base_elements']:
                            # condition for base element to be added -> doesnt collide with values and is inside the area identified as line item area or on the right of it
                            if not g.is_inside(el) and not g.collides_with(el) and el.x0 < g.x0 and (
                                    chosen_bordering_area[g_index]['overlapping_elements_area'].x0 <= el.x0 or
                                    chosen_bordering_area[g_index]['overlapping_elements_area'].is_inside(el) or
                                    chosen_bordering_area[g_index]['overlapping_elements_area'].h_overlap(el)):
                                # add li object
                                if row_index not in line_item_helper:
                                    line_item_helper[row_index] = LineItem(el)
                                else:
                                    # add to li
                                    line_item_helper[row_index].add_el(el)
                        # check if something was found, otherwise try with actual elements instead of base_elements
                        if row_index not in line_item_helper:
                            for el in row['list']:
                                # condition for base element to be added -> doesnt collide with values and is inside the area identified as line item area or on the right of it
                                if not g.is_inside(el) and not g.collides_with(el) and el.x0 < g.x0 and (
                                        chosen_bordering_area[g_index]['overlapping_elements_area'].x0 <= el.x0 or
                                        chosen_bordering_area[g_index]['overlapping_elements_area'].is_inside(el) or
                                        chosen_bordering_area[g_index]['overlapping_elements_area'].h_overlap(el)):
                                    # add li object
                                    if row_index not in line_item_helper:
                                        line_item_helper[row_index] = LineItem(el)
                                    else:
                                        # add to li
                                        line_item_helper[row_index].add_el(el)

            g.line_items = list(line_item_helper.values())

    def _split_table_if_needed(self, table: ExtractedTable) -> List[ExtractedTable]:

        output_list = []

        split_table = False
        for k, li in enumerate(table.items):
            if k > 0:
                if abs(table.items[k - 1].el.y0 - li.el.y1) > self.config.separate_table_distance:
                    split_table = True
                    table_one = ExtractedTable(table.items[0:k], table.g_index)
                    output_list.append(table_one)
                    # table 2
                    table_two = ExtractedTable(table.items[k:], table.g_index)
                    output_list += self._split_table_if_needed(table_two)
                    break
        if not split_table:
            output_list = [table]

        return output_list

    def extract_tables(self, min_rows: int = 1, min_cols: int = 1) -> List[ExtractedTable]:

        self._find_rows()
        self._find_columns_numeric()
        self._add_unbroken_areas()
        self._detect_tables()
        self._detect_line_items()

        tables = []

        for g_index, g in enumerate(self.groups):

            if len(g.elements) < min_cols:
                continue

            # sort elements
            g.elements = sorted(g.elements, key=lambda x: x.x1_el)

            value_grid = {row_index: [None for _ in range(0, len(g.elements))] for row_index in g.elements_by_row().keys()}

            for k, area in enumerate(g.elements):

                area.elements = sorted(area.elements, key=lambda x: (x.x0, -x.y1))

                for el in area.elements:

                    # find bounding el in row
                    bounding_el = None
                    for bounding_el_row in self.rows[el.row_index]['base_elements']:
                        bounding_identical = bounding_el_row.is_identical(el)
                        if bounding_identical or el.in_list(bounding_el_row.elements):
                            bounding_el = bounding_el_row
                            break

                    if bounding_el is not None:

                        # if bounding el contains more than one number cell, take the original element instead of bounding
                        if len([None for x in bounding_el.elements if is_number_cell(x.text)]) > 1:
                            bounding_el = el

                        # check that element is really in grid
                        if bounding_el.row_index not in value_grid:
                            continue

                        if value_grid[bounding_el.row_index][k] is None:
                            value_grid[bounding_el.row_index][k] = bounding_el
                        else:
                            if value_grid[bounding_el.row_index][k].text == bounding_el.text:
                                continue
                            else:
                                # merge elements
                                value_grid[bounding_el.row_index][k] = copy.deepcopy(value_grid[bounding_el.row_index][k])
                                value_grid[bounding_el.row_index][k].merge(bounding_el)

            # create table elements
            final_table = ExtractedTable(g.line_items, g_index)

            # fill empty line items
            final_table.fill_empty_li(value_grid)

            # separate table if li's too far away from each other
            separate_final_tables = self._split_table_if_needed(final_table)

            # filter out empty columns
            separate_final_tables = filter_out_empty_columns(separate_final_tables, min_cols)

            tables += separate_final_tables

        # sort
        tables = sorted(tables, key=lambda x: -x.table_area.y1)

        # discard tables that collide with another table (take table with more rows/columns)
        to_del = set()
        for k, table in enumerate(tables):
            # discard tables with not enough rows
            if len(table.items) < min_rows:
                to_del.add(k)
            if k in to_del:
                continue
            for kk in range(k+1, len(tables)):
                if kk in to_del:
                    continue
                if table.table_area.collides_with(tables[kk].table_area):
                    table1_score = table.num_rows * table.num_cols
                    table2_score = tables[kk].num_rows * tables[kk].num_cols
                    to_discard = k if table1_score < table2_score else kk
                    to_del.add(to_discard)

        for k in range(len(tables) - 1, -1, -1):
            if k in to_del:
                del (tables[k])

        return tables

    def extract_text_and_tables(self, min_rows: int = 2, min_cols: int = 1, **kwargs) -> List[ExtractedPdfElement]:

        if min_cols < 1 or min_rows < 1:
            raise Exception("a table needs at least one column and one row")

        # launch table recognition
        tables = self.extract_tables(min_rows, min_cols)

        all_elements = []
        all_extracted_elements: List[ExtractedPdfElement] = []

        for row in self.rows:
            for base_el in row['base_elements']:
                # check if element collides with a table
                in_table = False
                for t in tables:
                    if t.table_area.collides_with(base_el) or t.table_area.is_inside(base_el):
                        in_table = True
                        # add table now also to all elements
                        if t not in all_elements:
                            table_inserted = False
                            # CHECK FIRST IF TABLE WAS SPLIT, IF SO, SORT AFTER THE FIRST SPLIT TABLE
                            all_tables = [(k, x.g_index) for k, x in enumerate(all_elements) if
                                          isinstance(x, ExtractedTable)]
                            if t.g_index in [x[1] for x in all_tables]:
                                # find last occurrence of g_index
                                idx = None
                                for kk in range(len(all_tables) - 1, -1, -1):
                                    if all_tables[kk][1] == t.g_index:
                                        idx = all_tables[kk][0]
                                        break
                                if idx is not None:
                                    table_inserted = True
                                    all_elements.insert(idx + 1, t)
                                    all_extracted_elements.insert(idx + 1, t)
                            # add at the end if not inserted through previous g_index
                            if not table_inserted:
                                all_elements.append(t)
                                all_extracted_elements.append(t)
                        # check if element is inside line item area
                        if t.line_item_area.is_inside(base_el) or t.line_item_area.overlap_percent(base_el) > 0.8:
                            # check if element was used
                            if not t.line_item_area.contains(base_el):
                                t.add_to_items(base_el)
                        # check if element is inside value area
                        elif t.total_value_area.is_inside(base_el):
                            # check if element was placed
                            if base_el not in t.total_value_area.elements:
                                # try to place element
                                for col_idx, area in enumerate(t.value_areas):
                                    if area.collides_with(base_el) or area.is_inside(base_el):
                                        if base_el not in area.elements:
                                            # element is not placed yet, check if there is space
                                            if base_el.row_index not in area.all_row_indices:
                                                area.put_element(base_el)
                                                t.add_value(base_el, col_idx)
                                                break
                        else:
                            in_table = False
                        break
                # add text element
                if not in_table:
                    all_extracted_elements.append(ExtractedPdfElement(base_el.x0, base_el.x1, base_el.y0, base_el.y1, base_el))

        return all_extracted_elements
