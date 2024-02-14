import os
import re
import shutil
import tempfile
from decimal import Decimal
from subprocess import call
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from typing import List, Dict, Optional

years_abs_strings = []

IMG_BATCH_SIZE = 10

to_filter_numbers = re.compile(r'(\([^0-9 ]*\))|[^0-9A-Za-z/]')

month_periods = {
    3: [r'ytd mar', r'january( to |(| )-(| )|(| )–(| ))march', r'jan(u|)(\.|)( to |(| )-(| )|(| )–(| ))mar(\.|)',
        r'february( to |(| )-(| )|(| )–(| ))april', r'feb(r|)(\.|)( to |(| )-(| )|(| )–(| ))apr(\.|)',
        r'march( to |(| )-(| )|(| )–(| ))may', r'mar(\.|)( to |(| )-(| )|(| )–(| ))may(\.|)',
        r'april( to |(| )-(| )|(| )–(| ))june', r'apr(\.|)( to |(| )-(| )|(| )–(| ))jun(\.|)',
        r'may( to |(| )-(| )|(| )–(| ))july', r'may(\.|)( to |(| )-(| )|(| )–(| ))jul(\.|)',
        r'june( to |(| )-(| )|(| )–(| ))august', r'jun(\.|)( to |(| )-(| )|(| )–(| ))aug(\.|)',
        r'july( to |(| )-(| )|(| )–(| ))september', r'jul(\.|)( to |(| )-(| )|(| )–(| ))sep(t|)(\.|)',
        r'august( to |(| )-(| )|(| )–(| ))october', r'aug(\.|)( to |(| )-(| )|(| )–(| ))oct(\.|)',
        r'september( to |(| )-(| )|(| )–(| ))november', r'sep(t|)(\.|)( to |(| )-(| )|(| )–(| ))nov(\.|)',
        r'october( to |(| )-(| )|(| )–(| ))december', r'oct(\.|)( to |(| )-(| )|(| )–(| ))dec(\.|)',
        r'november( to |(| )-(| )|(| )–(| ))january', r'nov(\.|)( to |(| )-(| )|(| )–(| ))jan(u|)(\.|)',
        r'december( to |(| )-(| )|(| )–(| ))february', r'dec(\.|)( to |(| )-(| )|(| )–(| ))feb(r|)(\.|)'],
    6: [r'ytd jun', r'january( to |(| )-(| )|(| )–(| ))june', r'jan(u|)(\.|)( to |(| )-(| )|(| )–(| ))jun(\.|)',
        r'february( to |(| )-(| )|(| )–(| ))july', r'feb(r|)(\.|)( to |(| )-(| )|(| )–(| ))jul(\.|)',
        r'march( to |(| )-(| )|(| )–(| ))august', r'mar(\.|)( to |(| )-(| )|(| )–(| ))aug(\.|)',
        r'april( to |(| )-(| )|(| )–(| ))september', r'apr(\.|)( to |(| )-(| )|(| )–(| ))sep(t|)(\.|)',
        r'may( to |(| )-(| )|(| )–(| ))october', r'may(\.|)( to |(| )-(| )|(| )–(| ))oct(\.|)',
        r'june( to |(| )-(| )|(| )–(| ))november', r'jun(\.|)( to |(| )-(| )|(| )–(| ))nov(\.|)',
        r'july( to |(| )-(| )|(| )–(| ))december', r'jul(\.|)( to |(| )-(| )|(| )–(| ))dec(\.|)',
        r'august( to |(| )-(| )|(| )–(| ))january', r'aug(\.|)( to |(| )-(| )|(| )–(| ))jan(u|)(\.|)',
        r'september( to |(| )-(| )|(| )–(| ))february', r'sep(t|)(\.|)( to |(| )-(| )|(| )–(| ))feb(r|)(\.|)',
        r'october( to |(| )-(| )|(| )–(| ))march', r'oct(\.|)( to |(| )-(| )|(| )–(| ))mar(\.|)',
        r'november( to |(| )-(| )|(| )–(| ))april', r'nov(\.|)( to |(| )-(| )|(| )–(| ))apr(\.|)',
        r'december( to |(| )-(| )|(| )–(| ))may', r'dec(\.|)( to |(| )-(| )|(| )–(| ))may(\.|)'],
    9: [r'ytd sep', r'january( to |(| )-(| )|(| )–(| ))september',
        r'jan(u|)(\.|)( to |(| )-(| )|(| )–(| ))sep(t|)(\.|)', r'february( to |(| )-(| )|(| )–(| ))october',
        r'feb(r|)(\.|)( to |(| )-(| )|(| )–(| ))oct(\.|)', r'march( to |(| )-(| )|(| )–(| ))november',
        r'mar(\.|)( to |(| )-(| )|(| )–(| ))nov(\.|)', r'april( to |(| )-(| )|(| )–(| ))december',
        r'apr(\.|)( to |(| )-(| )|(| )–(| ))dec(\.|)', r'may( to |(| )-(| )|(| )–(| ))january',
        r'may(\.|)( to |(| )-(| )|(| )–(| ))jan(u|)(\.|)', r'june( to |(| )-(| )|(| )–(| ))february',
        r'jun(\.|)( to |(| )-(| )|(| )–(| ))feb(r|)(\.|)', r'july( to |(| )-(| )|(| )–(| ))march',
        r'jul(\.|)( to |(| )-(| )|(| )–(| ))mar(\.|)', r'august( to |(| )-(| )|(| )–(| ))april',
        r'aug(\.|)( to |(| )-(| )|(| )–(| ))apr(\.|)', r'september( to |(| )-(| )|(| )–(| ))may',
        r'sep(t|)(\.|)( to |(| )-(| )|(| )–(| ))may(\.|)', r'october( to |(| )-(| )|(| )–(| ))june',
        r'oct(\.|)( to |(| )-(| )|(| )–(| ))jun(\.|)', r'november( to |(| )-(| )|(| )–(| ))july',
        r'nov(\.|)( to |(| )-(| )|(| )–(| ))jul(\.|)', r'december( to |(| )-(| )|(| )–(| ))august',
        r'dec(\.|)( to |(| )-(| )|(| )–(| ))aug(\.|)'],
    12: [r'ytd dec', r'january( to |(| )-(| )|(| )–(| ))december', r'jan(u|)(\.|)( to |(| )-(| )|(| )–(| ))dec(\.|)',
         r'february( to |(| )-(| )|(| )–(| ))january', r'feb(r|)(\.|)( to |(| )-(| )|(| )–(| ))jan(u|)(\.|)',
         r'march( to |(| )-(| )|(| )–(| ))february', r'mar(\.|)( to |(| )-(| )|(| )–(| ))feb(r|)(\.|)',
         r'april( to |(| )-(| )|(| )–(| ))march', r'apr(\.|)( to |(| )-(| )|(| )–(| ))mar(\.|)',
         r'may( to |(| )-(| )|(| )–(| ))april', r'may(\.|)( to |(| )-(| )|(| )–(| ))apr(\.|)',
         r'june( to |(| )-(| )|(| )–(| ))may', r'jun(\.|)( to |(| )-(| )|(| )–(| ))may(\.|)',
         r'july( to |(| )-(| )|(| )–(| ))june', r'jul(\.|)( to |(| )-(| )|(| )–(| ))jun(\.|)',
         r'august( to |(| )-(| )|(| )–(| ))july', r'aug(\.|)( to |(| )-(| )|(| )–(| ))jul(\.|)',
         r'september( to |(| )-(| )|(| )–(| ))august', r'sep(t|)(\.|)( to |(| )-(| )|(| )–(| ))aug(\.|)',
         r'october( to |(| )-(| )|(| )–(| ))september', r'oct(\.|)( to |(| )-(| )|(| )–(| ))sep(t|)(\.|)',
         r'november( to |(| )-(| )|(| )–(| ))october', r'nov(\.|)( to |(| )-(| )|(| )–(| ))oct(\.|)',
         r'december( to |(| )-(| )|(| )–(| ))november', r'dec(\.|)( to |(| )-(| )|(| )–(| ))nov(\.|)']}
month_periods_list = []
for _, month_per in month_periods.items():
    month_periods_list += month_per
re_month_periods = re.compile("|".join(month_periods_list))


def page_size_db(page_size_extractor):
    return str(round(page_size_extractor[0])) + "," + str(round(page_size_extractor[1])) + "," + str(
        round(page_size_extractor[2])) + "," + str(round(page_size_extractor[3]))


def month_digit_to_str(month_digit):
    if month_digit == 1:
        return "january"
    elif month_digit == 2:
        return "february"
    elif month_digit == 3:
        return "march"
    elif month_digit == 4:
        return "april"
    elif month_digit == 5:
        return "may"
    elif month_digit == 6:
        return "june"
    elif month_digit == 7:
        return "july"
    elif month_digit == 8:
        return "august"
    elif month_digit == 9:
        return "september"
    elif month_digit == 10:
        return "october"
    elif month_digit == 11:
        return "november"
    elif month_digit == 12:
        return "december"
    else:
        return None


def contains_date(cell_str):
    date_formats = [r'\b([0123]|)\d[\.-\/]( |)(([0]|)\d|10|11|12)[\.-\/]( |)(([12]\d|)\d{2})\b',
                    r'\b(([12]\d|)\d{2})[-](([0]|)\d|10|11|12)[-]([0123]|)\d\b']

    month = None
    year = None
    for dk, pattern in enumerate(date_formats):
        d_s = re.search(pattern, cell_str)
        if d_s:
            date_matched = dk

            # date matched
            if date_matched == 0:
                # format: DD.MM.YYYY
                month = int(d_s.group(3))
                year = int(d_s.group(6))

                if len(d_s.group(6)) == 2:
                    if year > 40:
                        year = 1900 + year
                    else:
                        year = 2000 + year
            else:
                # format: YYYY-MM-DD
                month = int(d_s.group(3))
                year = int(d_s.group(1))

                if len(d_s.group(1)) == 2:
                    if year > 40:
                        year = 1900 + year
                    else:
                        year = 2000 + year
            break
    return month_digit_to_str(month), year


def cell_type(cell_str, clean=True):
    cell_str = str(cell_str).lower()

    period_strings = ["quarter", "year", "fy", "full", "financial year", "quarterly", "yearly", "annual",
                      "3m", "6m", "9m", "12m", "three", "six", "nine", "twelve", "month", "half",
                      "first", "second", "third", "fourth", "1st", "2nd", "3rd", "4th",
                      "q1", "q2", "q3", "q4", "i", "ii", "iii", "h1", "h2", "1q", "2q", "3q", "4q", "1h", "2h", "ytd"]

    month_strings = ["january", "jan", "janu", "february", "feb", "febr", "march", "mar", "april", "apr", "may", "june",
                     "jun", "july", "jul", "august", "aug", "september", "sep", "sept", "october",
                     "oct", "november", "nov", "december", "dec"]

    unit_strings = ["thousand", "million", "billion", "bn", "1000", "1000000", "unit", "units", "percent", "per cent",
                    "%change", "%", "in%", "change%", "000"]

    currency_strings = ['afn', 'eur', 'euro', 'lek', 'dzd', 'dinar', 'usd', 'us', 'dollar', 'aoa', 'kwanza', 'xcd',
                        'ars', 'peso', 'amd', 'dram', 'awg', 'aruban', 'florin', 'aud', 'azn', 'manat',
                        'bsd', 'bhd', 'bdt', 'taka', 'bbd', 'byn', 'ruble', 'bzd', 'xof', 'cfa', 'franc', 'bceao',
                        'bmd', 'inr', 'rupee', 'btn', 'ngultrum', 'bob', 'bov', 'mvdol', 'bam', 'mark',
                        'bwp', 'pula', 'nok', 'krone', 'brl', 'real', 'bnd', 'bgn', 'lev', 'bif', 'cve', 'cabo',
                        'verde', 'escudo', 'khr', 'riel', 'xaf', 'beac', 'cad', 'kyd', 'clp', 'clf', 'cny',
                        'yuan', 'renminbi', 'cop', 'cou', 'valor', 'kmf', 'cdf', 'nzd', 'crc', 'colon', 'hrk', 'kuna',
                        'cup', 'cuc', 'ang', 'guilder', 'czk', 'czech', 'koruna', 'dkk', 'djf', 'dop',
                        'egp', 'pound', 'svc', 'el', 'ern', 'nakfa', 'etb', 'birr', 'fkp', 'fjd', 'fiji', 'xpf', 'cfp',
                        'gmd', 'lari', 'ghs', 'cedi', 'gip', 'gtq', 'gbp', 'sterling', 'gnf', 'gyd',
                        'htg', 'hnl', 'hkd', 'huf', 'forint', 'isk', 'krona', 'idr', 'rupiah', 'irr', 'rial', 'iqd',
                        'iraqi', 'ils', 'sheqel', 'jmd', 'jpy', 'yen', 'jod', 'kzt', 'kes', 'shilling',
                        'kpw', 'won', 'krw', 'kwd', 'kgs', 'som', 'lak', 'kip', 'lbp', 'lsl', 'loti', 'zar', 'rand',
                        'lrd', 'lyd', 'chf', 'mop', 'pataca', 'mkd', 'denar', 'mga', 'mwk', 'myr', 'mvr',
                        'mru', 'mur', 'mxn', 'mxv', 'mdl', 'leu', 'mnt', 'mad', 'mzn', 'mmk', 'kyat', 'nad', 'npr',
                        'nio', 'oro', 'ngn', 'naira', 'omr', 'pkr', 'pab', 'pgk', 'kina', 'pyg', 'pen',
                        'sol', 'php', 'pln', 'zloty', 'qar', 'ron', 'rub', 'rwf', 'shp', 'wst', 'stn', 'dobra', 'sar',
                        'riyal', 'rsd', 'scr', 'sll', 'sgd', 'sbd', 'sos', 'ssp', 'south', 'lkr', 'sdg',
                        'srd', 'szl', 'sek', 'che', 'wir', 'chw', 'syp', 'twd', 'tjs', 'tzs', 'thb', 'baht', 'top',
                        'ttd', 'tnd', 'lira', 'tmt', 'ugx', 'uah', 'aed', 'uae', 'usn', 'uyu', 'uyi', 'uyw',
                        'uzs', 'vuv', 'vatu', 'ves', 'vnd', 'yer', 'zmw', 'zwl']

    mixed_unit_currency = ["eurk", "eurm", "usdk", "usdm", "$m", "€m", "£m", "¥m", "$k", "€k", "£k", "¥k", "$bn", "€bn",
                           "£bn", "¥bn", "dkkk", "kdkk", "dkkm", "mdkk"]

    currency_signs = ("$", "€", "£", "¥")

    output = []
    if clean:
        components = list(filter(lambda x: x != "", re.sub('[^A-Za-z0-9%$€£¥]', ' ', cell_str).split(" ")))
    else:
        components = list(filter(lambda x: x != "", cell_str.split(" ")))

    # check if cell contains a date
    month_date_match, year_date_match = contains_date(cell_str)

    to_append = []
    check_for_period = False
    if is_number_cell(cell_str):
        for ck, c in enumerate(components):

            if c not in currency_signs:
                if c.endswith(currency_signs):
                    currency = c[-1]
                    c = c[:-1]
                    components[ck] = c
                    components.append(currency)
                if c.startswith(currency_signs):
                    currency = c[0]
                    c = c[1:]
                    components[ck] = c
                    components.append(currency)

            if c in years_abs_strings:
                output.append("time-year")
            elif c in unit_strings or (c.endswith("s") and c[:-1] in unit_strings):
                output.append("num-unit")
            elif c in currency_strings or (c.endswith("s") and c[:-1] in currency_strings):
                output.append("num-currency")
            elif c in currency_signs or c.endswith(currency_signs) or c.startswith(currency_signs):
                output.append("num-currency")
            else:
                output.append("num-value")
    else:
        for ck, c in enumerate(components):

            if c in years_abs_strings:
                output.append("time-year")
            elif c in period_strings or (c.endswith("s") and c[:-1] in period_strings):
                output.append("time-period")
            elif c in month_strings:
                check_for_period = True
                output.append("time-month")
            elif c in unit_strings or (c.endswith("s") and c[:-1] in unit_strings):
                output.append("num-unit")
            elif c in currency_strings or (c.endswith("s") and c[:-1] in currency_strings):
                output.append("num-currency")
            elif c in currency_signs:
                output.append("num-currency")
            elif c in mixed_unit_currency:
                output.append("num-currency")
                output.append("num-unit")
                to_append.append((ck, c))
            else:
                output.append("other")

    for t in to_append:
        components.insert(t[0], t[1])

    if month_date_match is not None:
        # add dates from date match
        if "time-month" not in output:
            output.append("time-month")
            components.append(month_date_match)
        if "time-year" not in output:
            output.append("time-year")
            components.append(year_date_match)

    if check_for_period:
        # check if str contains month-period
        if len(re.findall(re_month_periods, cell_str)) > 0:
            output.append("time-period")
            components.append(cell_str)

    return output, components


def is_date_cell(cell_str):
    if re.search(
            r'(\b([0123]|)\d[\.-\/]( |)(([0]|)\d|10|11|12)[\.-\/]( |)(([12]\d|)\d{2})\b)|(\b(([12]\d|)\d{2})[-](([0]|)\d|10|11|12)[-]([0123]|)\d\b)',
            cell_str):
        return True
    return False


def space_separator_thousands(cell_str):
    if re.search(r'\b[0-9]{1,3} [0-9]{3}\b', cell_str):
        return True
    return False


def letter_len(string):
    string = re.sub('[^A-Za-z]', '', string)
    return len(string)


def words_contained(cell_str, lower=False):
    if lower:
        cell_str = cell_str.lower()
    return list(filter(lambda x: x != "", re.sub('[^A-Za-z0-9%$€£¥]', ' ', cell_str).split(" ")))


def is_year_cell(cell_str):
    pieces = words_contained(cell_str)
    for p in pieces:
        if p in years_abs_strings:
            return True
    return False


def comma_dot_separator_thousands(cell_str):
    if re.search(r'\b[0-9]{1,3}[,.][0-9]{3}\b', cell_str):
        return True
    return False


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
