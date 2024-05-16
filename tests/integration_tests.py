from pdf_reader.extract import get_elements_from_pdf, ExtractedTable


def test_rows_cols1():
    pages = get_elements_from_pdf("./files/form10k20231230.pdf")
    tables = [x for x in pages[0].paragraphs if isinstance(x, ExtractedTable)]
    assert len(tables) == 1
    assert tables[0].num_cols == 2
    assert tables[0].num_rows == 47


def test_rows_cols_bayer():
    pages = get_elements_from_pdf("./files/bayer1.pdf")
    assert len(pages) == 50
    # page 3
    tables = [x for x in pages[2].paragraphs if isinstance(x, ExtractedTable)]
    assert len(tables) == 1
    assert tables[0].num_cols == 7
    assert 36 <= tables[0].num_rows <= 37
    # page 6
    tables = [x for x in pages[5].paragraphs if isinstance(x, ExtractedTable)]
    assert len(tables) == 2
    assert tables[0].num_cols == 8
    assert 10 <= tables[0].num_rows <= 12
    assert tables[1].num_cols == 8
    assert 9 <= tables[1].num_rows <= 10
    # page 9
    tables = [x for x in pages[8].paragraphs if isinstance(x, ExtractedTable)]
    assert len(tables) >= 1
    assert tables[0].num_cols == 8
    assert 22 <= tables[0].num_rows <= 24


def test_rows_cols_allianz():
    pages = get_elements_from_pdf("./files/q111_interimreport.pdf")
    assert len(pages) == 95
    # page 3
    tables = [x for x in pages[2].paragraphs if isinstance(x, ExtractedTable)]
    assert len(tables) == 1
    assert tables[0].num_cols == 3
    assert 33 <= tables[0].num_rows <= 35
    # page 4
    tables = [x for x in pages[3].paragraphs if isinstance(x, ExtractedTable)]
    assert len(tables) >= 1
    assert 5 <= tables[0].num_rows <= 7
    assert tables[0].num_cols == 3
    assert "Total revenues" in [x.caption for x in tables[0].items]
