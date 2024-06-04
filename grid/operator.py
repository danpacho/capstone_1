import copy
import sys
import decoder as gd  # pylint: disable=import-error
import encoder as ge  # pylint: disable=import-error
import csv
import numpy as np
import matplotlib.pyplot as plt


def csv_reader(csv_name="dpset.csv"):
    imported_csv = np.load(csv_name, encoding="ASCII")
    dp_list = imported_csv.values.tolist()

    return dp_list


def csv_saver(my_list, file_name):
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(my_list)


def dp_existed_checker(new_list, existing_lists):
    return any(new_list == existing_list for existing_list in existing_lists)


def dp_add(vent_opened, dp_list, trashIO=True, monitorIO=False):
    if vent_opened not in range(0, 341):
        sys.exit("0~340사이 열린 벤트수 작성")

    io = False
    while io is False:
        generated = ge.dpnumgen(vent_opened, trashIO)
        if dp_existed_checker(generated, dp_list):
            io = False
            if monitorIO:
                print("새로 추가된 리스트는 기존 이중 리스트에 있습니다.")
        else:
            io = True
            dp_list.append(generated)
            if monitorIO:
                print("새로 추가된 리스트는 기존 이중 리스트에 없습니다.")

    if monitorIO:
        gd.plotter(generated)
        print(openCounter(generated))

    return dp_list


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


def dataset_analysis(dp_list):
    dp_type_counter = np.zeros(341)

    for k in range(0, len(dp_list)):
        opened = openCounter(dp_list[(k)])

        dp_type_counter[opened] = dp_type_counter[opened] + 1

    total_data_counted = 0

    for i in range(0, len(dp_type_counter)):
        total_data_counted = int(total_data_counted + dp_type_counter[i])
        print(i, ":", dp_type_counter[i])

    plt.plot(dp_type_counter)
    plt.show()

    return dp_type_counter
