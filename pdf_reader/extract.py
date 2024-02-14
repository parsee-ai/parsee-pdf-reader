from typing import *
import math
import pypdf

from pdf_reader.custom_dataclasses import *
from pdf_reader.converter import get_pdf_pages


def relative_areas_to_area_predictions(relative_areas: List[RelativeAreaPrediction], page_width: int, page_height: int) -> List[AreaPrediction]:

    return sorted([AreaPrediction(x, page_width, page_height, k) for k, x in enumerate(relative_areas) if x.prob > 0.5], key=lambda x: -x.prob)


def make_paragraphs(elements: List[ExtractedPdfElement], config: PdfReaderConfig, element_areas: Union[None, List[AreaPrediction]], all_text: Union[None, str]) -> List[ExtractedPdfElement]:
    elements = sorted(elements, key=lambda x: x.y1, reverse=True)
    current_groups: List[PdfParagraph] = []
    all_groups: List[ExtractedPdfElement] = []
    GROUP_TOLERANCE = 10
    element_areas = [] if element_areas is None else element_areas

    natural_text = all_text.replace("\n", " ") if all_text is not None else None

    figures = [x for x in element_areas if x.class_value == "figure"]

    for el in elements:
        # check if el is contained in some detected area
        el.dict_el["in_area"] = None
        for area in element_areas:
            if area.is_inside(el):
                el.dict_el["in_area"] = area
                break
        if el.dict_el["in_area"] in figures:
            # handle figures separately
            continue
        if el.dict_el["type"] == "em":
            added_to_group = False
            # check if element can be added to a current group
            for n in range(len(current_groups) - 1, -1, -1):
                if current_groups[n].y0 - el.y1 > GROUP_TOLERANCE:
                    del current_groups[n]
                elif current_groups[n].y0 - el.y1 <= config.LINE_BREAK_MAX_DISTANCE:
                    # check if el can be added to group
                    add_to_group = False
                    if current_groups[n].h_inside(el) or current_groups[n].h_overlap(el):
                        add_to_group = True
                    elif current_groups[n].x_distance_to(el) <= config.SPACE_MAX_DISTANCE and el.dict_el["in_area"] is not None and el.dict_el["in_area"].class_id in current_groups[n].area_group_ids:
                        add_to_group = True
                    elif natural_text is not None and current_groups[n].x_distance_to(el) <= config.SPACE_MAX_DISTANCE and (((current_groups[n].elements[-2].get_text() + " ") if len(current_groups[n].elements) > 1 else "") + current_groups[n].elements[-1].get_text()+" "+el.get_text() in natural_text):
                        add_to_group = True
                    if add_to_group:
                        current_groups[n].add_el(el)
                        added_to_group = True
                        break
            # create new group if not added to existing one
            if not added_to_group:
                p = PdfParagraph([el], config)
                current_groups.append(p)
                all_groups.append(p)
        else:
            all_groups.append(el)

    for figure in figures:
        all_groups.append(ExtractedFigure(figure.x0, figure.x1, figure.y0, figure.y1))

    return list(sorted(all_groups, key=lambda el: (math.floor(el.x0 / 150), -(math.floor(el.y1 / 40)))))


# returns a list of elements (text, tables) from a PDF
def get_elements_from_pdf(pdf_path, detected_areas: Union[None, Dict[int, List[RelativeAreaPrediction]]] = None) -> List[ExtractedPage]:

    pages = get_pdf_pages(pdf_path)
    pypdf_reader = pypdf.PdfReader(pdf_path)

    all_pages = []

    for page_index, p in enumerate(pages):
        page_elements = p.extract_text_and_tables()
        areas = None
        if detected_areas is not None and page_index in detected_areas:
            areas = relative_areas_to_area_predictions(detected_areas[page_index], p.page_size.width(), p.page_size.height())
        try:
            pypdf_text = pypdf_reader.pages[page_index].extract_text()
        except Exception:
            pypdf_text = None
        paragraphs = make_paragraphs(page_elements, PdfReaderConfig(20, 10, 6), areas, pypdf_text)
        all_pages.append(ExtractedPage(page_index, p.page_size, page_elements, paragraphs))

    return all_pages
