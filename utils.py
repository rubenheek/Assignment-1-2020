import csv


def get_file_rows(filepath):
    '''
    Gives the data rows (ignoring the header) of the csv file corresponding to the given file path.
    '''
    with open(filepath, "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        return rows[1:]


def get_num_file_rows(filepath):
    '''
    Gives the number of data rows (ignoring the header) of the csv file corresponding to the given file path.
    '''
    with open(filepath, "r") as csvfile:
        rows = csvfile.readlines()
        return len(rows) - 1


def write_file(filepath, header, rows):
    '''
    Writes the given header row and data rows to the csv file corresponding to the given file path.
    '''
    with open(filepath, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
