import sys
import os
import numpy as np
import copy

def plotter(lst):

    num_list = copy.deepcopy(lst)
    if len(num_list)!=20:
        sys.exit('배열 길이 20 맞출 것')

    print("\n\n")
    print("        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20")
    print("\n\n")

    for i in range(0,5):
        print((20-i), "    ", end =" ")
        if i>10:
            print("", end =" ")
        row_data = num_list[(19-i)]
        print("        ", end =" ")
        for j in range(0,3):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        for j in range(3,17):
            if ((row_data-2**(19-j))>=0):
                print("O ", end =" ")
                row_data = (row_data-2**(19-j))
            else:
                print(". ", end =" ")
        for j in range(17,20):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")
        print("        ", end =" ")
        print("    ", (20-i))

    for i in range(5,10):
        print((20-i), "    ", end =" ")
        if i>10:
            print("", end ="     ")
        row_data = num_list[(19-i)]
        print("     ", end =" ")
        for j in range(0,2):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        for j in range(2,18):
            if ((row_data-2**(19-j))>=0):
                print("O ", end =" ")
                row_data = (row_data-2**(19-j))
            else:
                print(". ", end =" ")
        for j in range(18,20):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")
        print("     ", end =" ")
        print("    ", (20-i))

    for i in range(10,15):
        print((20-i), "    ", end =" ")
        if i>10:
            print("", end =" ")
        print("", end ="   ")
        row_data = num_list[(19-i)]
        for j in range(0,1):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        for j in range(1,19):
            if ((row_data-2**(19-j))>=0):
                print("O ", end =" ")
                row_data = (row_data-2**(19-j))
            else:
                print(". ", end =" ")
        for j in range(19,20):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")
        print("", end =" ")
        print("      ", (20-i))

    for i in range(15,20):
        print((20-i), "    ", end =" ")
        if i>10:
            print("", end =" ")
        row_data = num_list[(19-i)]
        for j in range(0,20):
            if ((row_data-2**(19-j))>=0):
                print("O ", end =" ")
                row_data = (row_data-2**(19-j))
            else:
                print(". ", end =" ")
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")
        print("    ", (20-i))

    print("\n\n")
    print("        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20")
    print("\n\n")

    return

def commander(lst):
    Contact_data = np.loadtxt("Contact_Bodies_Coordinates_System.csv", delimiter=',')
    Target_data = np.loadtxt("Target_Bodies_Coordinates_System.csv", delimiter=',')
    Contact_Saved = []
    Target_Saved = []

    num_list = copy.deepcopy(lst)
    if len(num_list)!=20:
        sys.exit('배열 길이 20 맞출 것')

    for i in range(0,5):
        row_data = num_list[(19-i)]
        for j in range(0,3):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        for j in range(3,17):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
                Contact_Saved.append(Contact_data[19-i][j])
                Target_Saved.append(Target_data[19-i][j])
        for j in range(17,20):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")

    for i in range(5,10):
        row_data = num_list[(19-i)]
        for j in range(0,2):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        for j in range(2,18):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
                Contact_Saved.append(Contact_data[19-i][j])
                Target_Saved.append(Target_data[19-i][j])
        for j in range(18,20):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")

    for i in range(10,15):
        row_data = num_list[(19-i)]
        for j in range(0,1):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        for j in range(1,19):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
                Contact_Saved.append(Contact_data[19-i][j])
                Target_Saved.append(Target_data[19-i][j])
        for j in range(19,20):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")

    for i in range(15,20):
        row_data = num_list[(19-i)]
        for j in range(0,20):
            if ((row_data-2**(19-j))>=0):
                row_data = (row_data-2**(19-j))
                Contact_Saved.append(Contact_data[19-i][j])
                Target_Saved.append(Target_data[19-i][j])
        if row_data != 0:
            print("\n", row_data)
            sys.exit("error")

    Contact_Saved = map(int, Contact_Saved)
    Target_Saved = map(int, Target_Saved)

    print("#region Details View Action")
    print("contact_region_109 = DataModel.GetObjectById(109)")
    print("selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)")
    print("selection.Ids = [", ', '.join(map(str, Contact_Saved)), "]")
    print("contact_region_109.SourceLocation = selection \n #endregion\n")

    print("#region Details View Action")
    print("selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)")
    print("selection.Ids = [", ', '.join(map(str, Target_Saved)), "]")
    print("contact_region_109.TargetLocation = selection \n #endregion\n\n")

    return
