# Parsee PDF Reader

This PDF reader was designed to overcome the common problems when trying to extract tables from PDFs.

We started initially with a focus on financial/numeric tables, so this is where this PDF reader works best for.

This is an early release, so we will be still making major changes.

## Installation

Recommended install with poetry: https://python-poetry.org/docs/

    poetry add parsee-pdf-reader

Alternatively:

    pip install parsee-pdf-reader

## Extracting Tables and Paragraphs

Extracting tables and paragraphs of text can be done in one line:

    from pdf_reader import get_elements_from_pdf
    elements = get_elements_from_pdf("FILE_PATH")

If you are processing a PDF that needs OCR but no elements or just very few are being returned, you can force OCR like this:
    
    elements = get_elements_from_pdf("FILE_PATH", force_ocr=True)

## Methodology
Combines pdfminer, pypdf and tesseract and augments them with the introduction of table elements, which are treated separately from the rest of the text. As a result, the output contains basically two types of elements: tables and text paragraphs. We believe this separation is important as otherwise the tabular information is not extracted very precisely and concepts such as columns and rows are usually lost.