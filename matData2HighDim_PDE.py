# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io


# load the data from matlab of .mat
def loadMatlabIdata(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_data2Biharmonic(dim=2, data_path=None):
    if dim == 2:
        testData_file = str(data_path) + '/' + str('testXY') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XY']
    elif dim == 3:
        testData_file = str(data_path) + '/' + str('testXYZ') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XYZ']
    elif dim == 4:
        testData_file = str(data_path) + '/' + str('testXYZS') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XYZS']
    elif dim == 5:
        testData_file = str(data_path) + '/' + str('testXYZST') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XYZST']
    return data2test


if __name__ == '__main__':
    mat_data_path = 'data2mat'
    mat_data = get_data2Biharmonic(dim=2, data_path=mat_data_path)
    print('shdshd')