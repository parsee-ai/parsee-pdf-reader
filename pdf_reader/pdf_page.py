from typing import *
import math
import copy

import numpy as np
from pdfminer.layout import LTTextBox, LTTextLine, LTChar, Rect

from pdf_reader.custom_dataclasses import Rectangle, ElementMiner, Area, TableGroup, LineItem, ValueItem, ExtractedTable, ExtractedPdfElement, PdfReaderConfig
from pdf_reader.helper import is_number_cell, space_separator_thousands, comma_dot_separator_thousands, letter_len, is_year_cell, cell_type, is_date_cell, words_contained


def filter_out_empty_columns(output_list, min_cols_numeric):
    # filter out columns which have only None values
    for table_index in range(len(output_list) - 1, -1, -1):
        table_dict = output_list[table_index]
        if len(table_dict['values']) > 0:
            for col_index in range(len(table_dict['values'][0].values) - 1, -1, -1):
                if len([li.values[col_index] for li in table_dict['values'] if
                        li.values[col_index].is_empty()]) == len(table_dict['values']):
                    # splice column
                    for li in table_dict['values']:
                        del (li.values[col_index])
        if len(table_dict['values']) == 0 or len(table_dict['values'][0].values) < min_cols_numeric:
            del (output_list[table_index])

    return output_list


def make_final_tables(table_dict):
    final_tables = []
    final_tables_indices = []

    start_index = None
    end_index = None

    for idx, li in enumerate(table_dict['values']):
        if li.is_valid and not li.is_separator:
            if start_index is None:
                start_index = idx
            end_index = idx
        elif li.is_separator and start_index is not None and end_index is not None:
            # split table
            final_tables_indices.append((start_index, end_index))
            start_index = None
            end_index = None

        # last item
        if idx == len(table_dict['values']) - 1 and start_index is not None and end_index is not None:
            final_tables_indices.append((start_index, end_index))

    for idx_tuple in final_tables_indices:
        # filter out non valid items
        final_tables.append({"g_index": table_dict['g_index'],
                             "values": [x for x in table_dict['values'][idx_tuple[0]:idx_tuple[1] + 1] if
                                        x.is_valid]})

    return final_tables


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
    elements_list: List[ElementMiner]
    non_text_elements: List[ElementMiner]

    def __init__(self, page_index: int, pdf_path: str, page_size_pdfminer: Rect, text_boxes: List[LTTextBox], config: PdfReaderConfig):

        self.page_index = page_index
        self.pdf_path = pdf_path
        self.config = config
        self._set_page_size(page_size_pdfminer)
        self._set_elements(text_boxes)

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

    def _set_elements(self, text_boxes: List[LTTextBox]):

        self.elements_list = []
        self.non_text_elements = []

        # first: check if page contains some space separated values and no other thousands separator
        sp = 0
        replace_space_separator = True
        for element in text_boxes:
            if isinstance(element, LTTextBox):
                for o in element._objs:
                    if isinstance(o, LTTextLine):
                        try:
                            text = o.get_text()
                        except Exception as e:
                            text = ""
                        text_stripped = text.strip()
                        if is_number_cell(text_stripped):
                            if space_separator_thousands(text_stripped):
                                sp += 1
                            elif comma_dot_separator_thousands(text_stripped):
                                replace_space_separator = False
                                break
        if sp == 0:
            replace_space_separator = False

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
                                if (t == " " or t == "\t" or t == ")" or break_now) and (
                                        not replace_space_separator or not (
                                        replace_space_separator and is_number_cell(
                                    text_stripped) and space_separator_thousands(
                                    text_stripped) and t == " " and kk > 0 and o._objs[
                                            kk - 1].get_text() != " ")):
                                    if t == ")":
                                        new_text += ")"
                                        new_x1 = c.x1
                                    if new_text != "" and new_x0 is not None and new_x1 is not None:
                                        self.elements_list.append(
                                            ElementMiner(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text,
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
                                        ElementMiner(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text,
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
                            ElementMiner(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text, has_bold=has_bold,
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
                        ElementMiner(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text, has_bold=has_bold,
                                     scale_multiplier=self.scale_multiplier))
                    # no need to reset variables as end is reached anyway
            else:
                # check if an element still has to be added
                if new_text != "" and new_x0 is not None and new_x1 is not None:
                    self.elements_list.append(
                        ElementMiner(x0=new_x0, x1=new_x1, y0=new_y0, y1=new_y1, text=new_text, has_bold=has_bold,
                                     scale_multiplier=self.scale_multiplier))
                    # reset variables
                    new_text = ""
                    new_x0 = None
                    new_x1 = None
                    new_y0 = None
                    new_y1 = None
                self.non_text_elements.append(
                    ElementMiner(x0=element.x0, x1=element.x1, y0=element.y0, y1=element.y1, text="",
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

    def _get_bounding_text_area(self, element, col_elements_exclude, ref_x0=None, ref_x1=None, text=None, handled=None,
                               has_bold=None):

        if ref_x0 is None:
            ref_x0 = element.x0
        if ref_x1 is None:
            ref_x1 = element.x1
        if text is None:
            text = element.text
        if handled is None:
            handled = [element]
        if has_bold is None:
            has_bold = element.has_bold

        row = self.rows[element.row_index]

        for el in row['list']:
            if not element.is_identical(el) and not el.in_list(col_elements_exclude) and not el.in_list(handled):
                if abs(el.x0 - ref_x1) <= self.config.SPACE_MAX_DISTANCE and el.x1 > ref_x1:
                    ref_x1 = el.x1
                    text += " " + el.text
                    handled.append(el)
                    if el.has_bold:
                        has_bold = True
                    return self._get_bounding_text_area(element, col_elements_exclude, ref_x0, ref_x1, text, handled,
                                                       has_bold)
                if abs(ref_x0 - el.x1) <= self.config.SPACE_MAX_DISTANCE and el.x0 < ref_x0:
                    ref_x0 = el.x0
                    text = el.text + " " + text
                    handled.append(el)
                    if el.has_bold:
                        has_bold = True
                    return self._get_bounding_text_area(element, col_elements_exclude, ref_x0, ref_x1, text, handled,
                                                       has_bold)

        output_el = ElementMiner(ref_x0, ref_x1, element.y0, element.y1, text, element.row_index, has_bold)

        return output_el, handled

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
                    new_el, handled = self._get_bounding_text_area(el, handled)
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
                bounding_text, _ = self._get_bounding_text_area(el, all_col_items)
                if is_number_cell(bounding_text.text) or len(bounding_text.text) <= self.config.text_max_number_col:
                    el.bounding_text = bounding_text
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
                        if is_number_cell_check and not is_year_cell(all_elements[k].text):
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

        # groups that have some text aligned to the left are a VALID table
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
        scoring_weights = {"words": 5, "distance": 1, "completeness": 5}
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

    def _split_table_if_needed(self, table_dict):

        output_list = []

        table_dict['values'] = sorted(table_dict['values'], key=lambda li_obj: li_obj.el.row_index)
        split_table = False
        for k, li in enumerate(table_dict['values']):
            if k > 0:
                if abs(table_dict['values'][k - 1].el.y0 - li.el.y1) > self.config.separate_table_distance:
                    split_table = True
                    table_one = {"g_index": table_dict['g_index'], "values": table_dict['values'][0:k]}
                    output_list.append(table_one)
                    # table 2
                    table_dict['values'] = table_dict['values'][k:]
                    output_list += self._split_table_if_needed(table_dict)
                    break
        if not split_table:
            output_list = [table_dict]

        return output_list

    def _has_value_separator(self, li):

        bot = False
        top = False

        if "........" in li.el.text or "________" in li.el.text:
            bot = True

        # check for separator above/below values
        if len(li.values) > 0:
            value_el = li.first_non_empty_value()
            if value_el is not None:
                el = value_el.el
                for non_text_el in self.non_text_elements:
                    # take only real lines
                    if non_text_el.height() <= self.config.line_max_height and el.h_overlap(non_text_el):
                        if el.y0 > non_text_el.y0 and (abs(el.y0 - non_text_el.y0) <= self.config.SPACE_MAX_DISTANCE or abs(
                                el.y0 - non_text_el.y1) <= self.config.SPACE_MAX_DISTANCE):
                            bot = True
                        if el.y0 < non_text_el.y0 and (abs(el.y1 - non_text_el.y0) <= self.config.SPACE_MAX_DISTANCE or abs(
                                el.y1 - non_text_el.y1) <= self.config.SPACE_MAX_DISTANCE):
                            top = True
                        if bot and top:
                            return top, bot

        # check also for line item itself
        for non_text_el in self.non_text_elements:
            # take only real lines
            if non_text_el.height() <= self.config.line_max_height and li.el.h_overlap(non_text_el):
                if li.el.y0 > non_text_el.y0 and (abs(li.el.y0 - non_text_el.y0) <= self.config.SPACE_MAX_DISTANCE or abs(
                        li.el.y0 - non_text_el.y1) <= self.config.SPACE_MAX_DISTANCE):
                    bot = True
                if li.el.y0 < non_text_el.y0 and (abs(li.el.y1 - non_text_el.y0) <= self.config.SPACE_MAX_DISTANCE or abs(
                        li.el.y1 - non_text_el.y1) <= self.config.SPACE_MAX_DISTANCE):
                    top = True
                if bot and top:
                    break

        return top, bot

    def _consolidate_li_captions(self, g):

        line_item_area = None
        if len(g.line_items) > 0:
            line_item_area = Area(0, 0, 0, 0)
            line_item_area.init_with_elements([li.el for li in g.line_items])

        # consolidate line item captions eventually
        used_li_captions = []
        li_indices_to_delete = []
        for k, li in enumerate(g.line_items):
            if li.has_line_item_values():

                # check whether current li has separators above or below
                separator_above, separator_below = self._has_value_separator(li)

                if not separator_above:
                    # ABOVE current li
                    to_check_objs = []
                    if k > 0:
                        for a in range(k - 1, -1, -1):
                            if not g.line_items[a].has_line_item_values() and g.line_items[
                                a].el.row_index not in used_li_captions:
                                to_check_objs.append((a, g.line_items[a]))
                            else:
                                break

                    for kk, (li_idx, to_check_li) in enumerate(to_check_objs):
                        if li.el.y_distance_to(
                                to_check_li.el) < self.config.tolerance_top_bot_line_items and li.el.has_bold == to_check_li.el.has_bold:
                            # combined line item size has to (almost) exceed width of line item area
                            if to_check_li.el.width() + li.el.width() > line_item_area.width() * 0.9 and to_check_li.el.width() > line_item_area.width() * 0.5 and letter_len(
                                    to_check_li.caption) / len(to_check_li.caption) > 0.5:
                                # combine items
                                li.add_el(to_check_li.el)
                                used_li_captions.append(to_check_li.el.row_index)
                                li_indices_to_delete.append(li_idx)
                            else:
                                break
                        else:
                            break

                if not separator_below:
                    # BELOW current li
                    to_check_objs = []
                    if k < len(g.line_items) - 1:
                        for a in range(k + 1, len(g.line_items)):
                            if not g.line_items[a].has_line_item_values() and g.line_items[
                                a].el.row_index not in used_li_captions:
                                to_check_objs.append((a, g.line_items[a]))
                            else:
                                break

                    for kk, (li_idx, to_check_li) in enumerate(to_check_objs):
                        if li.el.y_distance_to(
                                to_check_li.el) < self.config.tolerance_top_bot_line_items and li.el.has_bold == to_check_li.el.has_bold:
                            # combined line item size has to (almost) exceed width of line item area
                            if to_check_li.el.width() + li.el.width() > line_item_area.width() * 0.9 and li.el.width() > line_item_area.width() * 0.5 and letter_len(
                                    li.caption) / len(li.caption) > 0.5:
                                # combine items
                                li.add_el(to_check_li.el)
                                used_li_captions.append(to_check_li.el.row_index)
                                li_indices_to_delete.append(li_idx)
                            else:
                                break
                        else:
                            break

        # delete all line items that were used
        g.line_items = [li for k, li in enumerate(g.line_items) if k not in li_indices_to_delete]

    def _find_meta_info(self, tables: List[ExtractedTable]):

        for table_index, table in enumerate(tables):

            # 1) look at all elements above table with certain distance
            if len(table.items) == 0:
                continue
            # start at first row above first value row
            max_row_index = table.items[0].el.row_index - 1
            if max_row_index < 0:
                continue
            max_y0 = table.items[0].el.y1 + self.config.table_header_max_height

            for row_index in range(max_row_index, -1, -1):

                row = self.rows[row_index]

                if row['y0_min'] > max_y0:
                    break

                part_of_table_prob = 1 - ((row['y0_min'] - table.items[0].el.y1) / self.config.table_header_max_height)

                for k, base_el in enumerate(row['base_elements']):
                    if base_el.x0 + self.config.table_sides_tolerance_x0 >= table.li_area.x0 and base_el.x1 - self.config.table_sides_tolerance_x1 <= \
                            table.value_areas[-1].x1:
                        # go through elements of base_element
                        for el in base_el.elements:
                            # check that element is not inside another table already
                            collides = False
                            for t2, table2 in enumerate(tables):
                                if t2 != table_index and (
                                        table2.table_area.is_inside(el) or table2.table_area.collides_with(el)):
                                    collides = True
                                    break
                            if collides:
                                continue

                            # adjust li area a bit to account for meta info that is slightly outside of text
                            li_area_copy = copy.deepcopy(table.li_area)
                            li_area_copy.x0 -= self.config.table_sides_tolerance_x0
                            c_overlap = el.h_overlap_percent([li_area_copy] + table.value_areas)

                            # condition for distance: either close to table OR close to an element that was already identified as meta
                            distance_condition = part_of_table_prob > 0.6 or (len(table.meta) > 0 and el.y0 - max(
                                [x.area.y1 for x in table.meta]) <= self.config.line_break_distance)

                            # check if element is overlapping entire table, make this check using base_el
                            c_overlap_table = table.table_area.h_overlap_percent([base_el])
                            overlap_condition = c_overlap_table[0] < 0.8

                            # add to table if distance condition is met
                            if distance_condition and overlap_condition:
                                table.add_meta_element(el, c_overlap)

            # refresh table areas to account for newly added meta elements
            table.set_table_area()

    def extract_tables(self, min_rows_numeric: int = 1, min_cols_numeric: int = 1) -> List[ExtractedTable]:

        self._find_rows()
        self._find_columns_numeric()
        self._add_unbroken_areas()
        self._detect_tables()
        self._detect_line_items()

        tables = []

        for g_index, g in enumerate(self.groups):

            if len(g.elements) < min_cols_numeric:
                continue

            # sort elements
            g.elements = sorted(g.elements, key=lambda x: x.x1_el)

            value_grid = {li.el.row_index: [None for _ in range(0, len(g.elements))] for li in g.line_items}

            # rows can be blacklisted if 2 distinct numeric values should be put in a single cell and the row is at the top, because this means it's probably part of the header still
            rows_blacklisted = []

            for k, area in enumerate(g.elements):

                area.elements = sorted(area.elements, key=lambda x: (x.x0, -x.y1))

                for el in area.elements:

                    # find bounding el in row
                    bounding_el = None
                    bounding_identical_check = False
                    for bounding_el_row in self.rows[el.row_index]['base_elements']:
                        bounding_identical = bounding_el_row.is_identical(el)
                        if bounding_identical or el.in_list(bounding_el_row.elements):
                            bounding_el = bounding_el_row
                            bounding_identical_check = bounding_identical
                            break

                    if bounding_el is not None and bounding_el.row_index not in rows_blacklisted:

                        # if bounding is not identical, double check that cell is really numeric
                        if not bounding_identical_check:
                            types, _ = cell_type(bounding_el.text)
                            if "num-value" not in types:
                                continue

                        # if bounding el contains more than one number cell, take the original element instead of bounding
                        if len([None for x in bounding_el.elements if is_number_cell(x.text)]) > 1:
                            bounding_el = el

                        # check: element needs line item
                        if bounding_el.row_index not in value_grid:
                            print("WARNING: value has no line item", "warning", bounding_el)
                            continue

                        # check: element can't contain a date
                        if is_date_cell(bounding_el.text):
                            continue

                        # check: bounding el can't be part of line item
                        cont = False
                        for li in g.line_items:
                            if li.el.row_index == bounding_el.row_index:
                                if bounding_el.in_list(li.el_list):
                                    cont = True
                                break
                        if cont:
                            continue

                        if value_grid[bounding_el.row_index][k] is None:
                            value_grid[bounding_el.row_index][k] = bounding_el
                        else:
                            if value_grid[bounding_el.row_index][k].text == bounding_el.text:
                                continue
                            # there is an element in place already, overwrite if new el is number and old not
                            if not is_number_cell(value_grid[bounding_el.row_index][k].text):
                                if is_number_cell(bounding_el.text):
                                    # overwrite
                                    value_grid[bounding_el.row_index][k] = bounding_el
                            else:
                                if is_number_cell(bounding_el.text):
                                    # 2 distinct numeric values in same cell -> doesn't work
                                    # check if we are at the top of the value grid, if so, blacklist this row as it's probably part of the header
                                    value_grid_rows = [x for x in list(sorted(value_grid.keys())) if
                                                       x not in rows_blacklisted]
                                    if bounding_el.row_index == value_grid_rows[0]:
                                        # set all cells to None in the row and blacklist row
                                        rows_blacklisted.append(bounding_el.row_index)
                                        for col_idx, _ in enumerate(g.elements):
                                            value_grid[bounding_el.row_index][col_idx] = None
                                    else:
                                        print("2 distinct numeric values for same cell", "warning",
                                                            (bounding_el, value_grid[bounding_el.row_index][k]))

            type_grid = {
                li.el.row_index: [cell_type(x.text)[0] if x is not None else None for x in value_grid[li.el.row_index]]
                for li in g.line_items}

            # assign values to line items
            final_table = {"g_index": g_index, "values": []}
            for row_index, val_list in value_grid.items():
                # find li
                chosen_li = None
                for li in g.line_items:
                    if li.el.row_index == row_index:
                        chosen_li = li
                        break
                # create empty line item if not found
                if chosen_li is None:
                    chosen_li = LineItem(None)
                chosen_li.assign_values(val_list, type_grid[row_index])
                final_table['values'].append(chosen_li)

            # separate table if meta row inbetween and filter out rows with no numeric values
            final_tables = make_final_tables(final_table)

            # separate table if li's too far away from each other
            separate_final_tables = []
            for table_dict in final_tables:
                separate_final_tables += self._split_table_if_needed(table_dict)

            # filter out empty columns
            separate_final_tables = filter_out_empty_columns(separate_final_tables, min_cols_numeric)

            # consolidate line item captions if possible
            self._consolidate_li_captions(g)

            for t in separate_final_tables:
                tables.append(t)

        # make table areas
        for t_index, table in enumerate(tables):
            li_area = Area(0, 0, 0, 0)
            li_area.init_with_elements([li.el for li in table['values']])
            table['li_area'] = li_area
            areas_temp = [[] for _ in table['values'][0].values]
            table['value_areas'] = []
            for li in table['values']:
                for col_index, val in enumerate(li.values):
                    areas_temp[col_index].append(val.el)
            for col in areas_temp:
                val_area = Area(0, 0, 0, 0)
                val_area.init_with_elements(col)
                table['value_areas'].append(val_area)

            # adjust value areas to extend full space they have, all but first col
            table['value_areas'] = sorted(table['value_areas'], key=lambda x: x.x0)
            for a in range(len(table['value_areas']) - 1, 0, -1):
                table['value_areas'][a].x0 = min(table['value_areas'][a].x0,
                                                 table['value_areas'][a - 1].x1 + self.config.SPACE_MAX_DISTANCE)
                table['value_areas'][a].y1 = max(table['value_areas'][a].y1, table['li_area'].y1)
                table['value_areas'][a].y0 = min(table['value_areas'][a].y0, table['li_area'].y0)
            # first column
            if len(table['value_areas']) > 1:
                table['value_areas'][0].x0 = min(table['value_areas'][0].x0,
                                                 table['value_areas'][0].x1 - table['value_areas'][1].width())
                table['value_areas'][0].y1 = max(table['value_areas'][0].y1, table['li_area'].y1)
                table['value_areas'][0].y0 = min(table['value_areas'][0].y0, table['li_area'].y0)
            # line items: close gap to first li
            if len(table['value_areas']) > 0:
                table['li_area'].x1 = max(table['li_area'].x1, table['value_areas'][0].x0 - self.config.SPACE_MAX_DISTANCE)

            # check in new value cells if some unaligned numeric values can be found, where cell is currently empty
            for li in table['values']:
                for col_index, val in enumerate(li.values):
                    if val.is_empty():
                        search_row_index = li.el.row_index
                        for el in self.rows[search_row_index]['base_elements']:
                            if table['value_areas'][col_index].is_inside(el) and is_number_cell(el.text):
                                li.values[col_index] = ValueItem(el)
                                break

        # convert to table objects
        extracted_tables = [ExtractedTable(table) for table in tables]

        # detect which elements are considered part of the table as meta elements
        self._find_meta_info(extracted_tables)

        # sort
        extracted_tables = sorted(extracted_tables, key=lambda x: -x.table_area.y1)

        # discard tables with not enough rows
        to_del = []
        for k, table in enumerate(extracted_tables):
            if len(table.items) < min_rows_numeric:
                to_del.append(k)

        for k in range(len(extracted_tables) - 1, -1, -1):
            if k in to_del:
                del (extracted_tables[k])

        return extracted_tables

    def extract_text_and_tables(self, min_rows_numeric: int = 1, min_cols_numeric: int = 1) -> List[ExtractedPdfElement]:
        # launch table recognition
        tables = self.extract_tables(min_rows_numeric, min_cols_numeric)

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
                            # CHECK FIRST IF TABLE WAS SPLIT, IF SO, SORT AFTER THE FIRST SPLITTED TABLE
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
                        # check if element is in meta info
                        if t.meta_area is not None and t.meta_area.is_inside(base_el):
                            continue
                        # check if element is inside line item area
                        if t.li_area.is_inside(base_el):
                            continue
                        # check if element is inside value area
                        if t.total_value_area.is_inside(base_el):
                            continue
                        # add element to other contained elements
                        if base_el not in t.other_contained_elements:
                            t.other_contained_elements.append(base_el)
                # add text element
                if not in_table:
                    all_extracted_elements.append(base_el.to_extracted_element())

        return all_extracted_elements
