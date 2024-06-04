import sys
import numpy as np
import copy
import random


def bool2code(lst, trashIO=True):
    bool_list = copy.deepcopy(lst)

    num_list = []
    if len(bool_list) != 340:
        sys.exit("배열 길이 340 맞출 것")

    for i in range(0, 340):
        if int(bool_list[i]) not in (0, 1):
            print(i)
            sys.exit("번째 bool값 오류")

    for k in range(0, 4):
        for i in range(0, 5):
            row_num = 0
            for j in range(0, k):
                if trashIO:
                    row_num = row_num + (2 ** (19 - j) * random.randint(0, 1))
            for j in range(k, 20 - k):
                n = int(
                    j
                    + 20 * i
                    + 20 * 5 * k
                    - 5 * 2 * (np.sum(np.arange(0, k)))
                    - (2 * k * i)
                    - k
                )
                if bool_list[n] == 1:
                    row_num = row_num + 2 ** (19 - j)
            for j in range(20 - k, 20):
                if trashIO:
                    row_num = row_num + (2 ** (19 - j) * random.randint(0, 1))
            if not (0 <= row_num <= 1048576):
                print(i, ",", row_num)
                sys.exit("줄 인코딩 오류")
            num_list.append(row_num)

    return num_list


def dpboolgen(vent_opened):
    vent = np.zeros(340)
    plot = 0
    if vent_opened not in range(0, 341):
        sys.exit("0~340사이 열린 벤트수 작성")

    # DP list 추가
    while plot < vent_opened:
        i = random.randint(0, 339)
        if vent[i] == int(0):
            vent[i] = int(1)
            plot = plot + 1

    return vent


def dpnumgen(open_num, trashIO=True):
    vents = dpboolgen(open_num)

    return bool2code(vents, trashIO)
