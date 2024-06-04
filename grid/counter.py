import copy
import sys


def openCounter(lst):
    if len(lst) != 20:
        sys.exit("배열 길이 20 맞출 것")
    counter = 0
    num_list = copy.deepcopy(lst)
    for i in range(0, 5):
        row_data = num_list[(19 - i)]
        for j in range(0, 3):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
        for j in range(3, 17):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
                counter = counter + 1
        for j in range(17, 20):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")

    for i in range(5, 10):
        row_data = num_list[(19 - i)]
        for j in range(0, 2):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
        for j in range(2, 18):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
                counter = counter + 1
        for j in range(18, 20):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")

    for i in range(10, 15):
        row_data = num_list[(19 - i)]
        for j in range(0, 1):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
        for j in range(1, 19):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
                counter = counter + 1
        for j in range(19, 20):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")
    for i in range(15, 20):
        row_data = num_list[(19 - i)]
        for j in range(0, 20):
            if (row_data - 2 ** (19 - j)) >= 0:
                row_data = row_data - 2 ** (19 - j)
                counter = counter + 1
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")

    return counter
