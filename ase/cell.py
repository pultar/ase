def same_cell(cell1, cell2):
    return ((cell1 is None) == (cell2 is None) and
            (cell1 is None or (cell1 == cell2).all()))
