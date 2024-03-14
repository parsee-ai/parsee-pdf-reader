# Parsee PDF Reader

This PDF reader was designed to overcome the common problems when trying to extract tables from PDFs.

We started initially with a focus on financial/numeric tables, so this is what this PDF reader works best for.

This is an early release, so we will be still making major changes.

## Installation

Recommended install with poetry: https://python-poetry.org/docs/

    poetry add parsee-pdf-reader

Alternatively:

    pip install parsee-pdf-reader

In order to use the OCR capabilities you also have to install tesseract:
[Install Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract) (additional info how to install the engine on Linux, Mac OSX and Windows). You must be able to invoke the tesseract command as 'tesseract'.
Note: in our testing we always used tesseract 5+, as that proved to be the most reliable. So for Linux you might have to build from source to get tesseract 5.

## Extracting Tables and Paragraphs

Extracting tables and paragraphs of text can be done in one line:

    from pdf_reader import get_elements_from_pdf
    elements = get_elements_from_pdf("FILE_PATH")

If you are processing a PDF that needs OCR but no elements or just very few are being returned, you can force OCR like this (replace the paths):
    
    elements = get_elements_from_pdf("FILE_PATH", force_ocr=True)

If you want to visualise the output from the extraction, you can run the following (replace the paths):

    from pdf_reader import visualise_pdf_output
    visualise_pdf_output("FILE_PATH", "OUTPUT_PATH")

This will save an image of each page with the detected tables and text marked in red.

## Methodology
Combines pdfminer, pypdf and tesseract and augments them with the introduction of table elements, which are treated separately from the rest of the text. As a result, the output contains basically two types of elements: tables and text paragraphs. We believe this separation is important as otherwise the tabular information is not extracted very precisely and concepts such as columns and rows are usually lost.