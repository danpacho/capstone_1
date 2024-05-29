import os
import sys
import numpy as np


def main():
    print("프로그램이 종료되었습니다.")


root = os.path.dirname(os.path.abspath(__file__))


Contact_data = np.loadtxt(
    f"{root}/data/Contact_Bodies_Coordinates_System.csv", delimiter=","
)
Target_data = np.loadtxt(
    f"{root}/data/Target_Bodies_Coordinates_System.csv", delimiter=","
)

Contact_Saved = []
Target_Saved = []

print("Design Point Matrix Direct Translator\n\n")

num_list = list(map(int, input().split(",")))
counter = 0

if len(num_list) != 20:
    sys.exit("배열 길이 20 맞출 것")

print("\n\n")
print("        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20")
print("\n\n")

for i in range(0, 5):
    print((20 - i), "    ", end=" ")
    if i > 10:
        print("", end=" ")
    row_data = num_list[(19 - i)]
    print("        ", end=" ")
    for j in range(0, 3):
        if (row_data - 2 ** (19 - j)) >= 0:
            row_data = row_data - 2 ** (19 - j)
    for j in range(3, 17):
        if (row_data - 2 ** (19 - j)) >= 0:
            print("O ", end=" ")
            row_data = row_data - 2 ** (19 - j)
            counter = counter + 1
            Contact_Saved.append(Contact_data[19 - i][j])
            Target_Saved.append(Target_data[19 - i][j])

        else:
            print(". ", end=" ")
    for j in range(17, 20):
        if (row_data - 2 ** (19 - j)) >= 0:
            row_data = row_data - 2 ** (19 - j)
    if row_data != 0:
        print("\n", row_data)
        sys.exit("error")
    print("        ", end=" ")
    print("    ", (20 - i))

for i in range(5, 10):
    print((20 - i), "    ", end=" ")
    if i > 10:
        print("", end="     ")
    row_data = num_list[(19 - i)]
    print("     ", end=" ")
    for j in range(0, 2):
        if (row_data - 2 ** (19 - j)) >= 0:
            row_data = row_data - 2 ** (19 - j)
    for j in range(2, 18):
        if (row_data - 2 ** (19 - j)) >= 0:
            print("O ", end=" ")
            row_data = row_data - 2 ** (19 - j)
            counter = counter + 1
            Contact_Saved.append(Contact_data[19 - i][j])
            Target_Saved.append(Target_data[19 - i][j])

        else:
            print(". ", end=" ")
    for j in range(18, 20):
        if (row_data - 2 ** (19 - j)) >= 0:
            row_data = row_data - 2 ** (19 - j)
    if row_data != 0:
        print("\n", row_data)
        sys.exit("error")
    print("     ", end=" ")
    print("    ", (20 - i))

for i in range(10, 15):
    print((20 - i), "    ", end=" ")
    if i > 10:
        print("", end=" ")
    print("", end="   ")
    row_data = num_list[(19 - i)]
    for j in range(0, 1):
        if (row_data - 2 ** (19 - j)) >= 0:
            row_data = row_data - 2 ** (19 - j)
    for j in range(1, 19):
        if (row_data - 2 ** (19 - j)) >= 0:
            print("O ", end=" ")
            row_data = row_data - 2 ** (19 - j)
            counter = counter + 1
            Contact_Saved.append(Contact_data[19 - i][j])
            Target_Saved.append(Target_data[19 - i][j])

        else:
            print(". ", end=" ")
    for j in range(19, 20):
        if (row_data - 2 ** (19 - j)) >= 0:
            row_data = row_data - 2 ** (19 - j)
    if row_data != 0:
        print("\n", row_data)
        sys.exit("error")
    print("", end=" ")
    print("      ", (20 - i))

for i in range(15, 20):
    print((20 - i), "    ", end=" ")
    if i > 10:
        print("", end=" ")
    row_data = num_list[(19 - i)]
    for j in range(0, 20):
        if (row_data - 2 ** (19 - j)) >= 0:
            print("O ", end=" ")
            row_data = row_data - 2 ** (19 - j)
            counter = counter + 1
            Contact_Saved.append(Contact_data[19 - i][j])
            Target_Saved.append(Target_data[19 - i][j])

        else:
            print(". ", end=" ")
    if row_data != 0:
        print("\n", row_data)
        sys.exit("error")
    print("    ", (20 - i))

print("\n\n")
print("        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20")
print("\n\n")


Contact_Saved = map(int, Contact_Saved)
Target_Saved = map(int, Target_Saved)

print("#region Details View Action")
print("contact_region_109 = DataModel.GetObjectById(109)")
print(
    "selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)"
)
print("selection.Ids = [", ", ".join(map(str, Contact_Saved)), "]")
print("contact_region_109.SourceLocation = selection \n #endregion\n")

print("#region Details View Action")
print(
    "selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)"
)
print("selection.Ids = [", ", ".join(map(str, Target_Saved)), "]")
print("contact_region_109.TargetLocation = selection \n #endregion\n\n")

print("벤트 격자수 : ", counter, "\n")


if __name__ == "__main__":
    main()
    input(
        "종료하려면 Enter 키를 누르세요..."
    )  # 프로그램이 끝난 후 사용자 입력을 기다림
