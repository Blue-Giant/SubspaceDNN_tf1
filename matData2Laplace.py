# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io


# load the data from matlab of .mat
def loadMatlabIdata(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_meshData2Laplace(equation_name=None, mesh_number=2):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = 'dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = 'dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = 'dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = 'dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = 'dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        assert (mesh_number == 6)
        test_meshXY_file = 'dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = 'dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_XY = loadMatlabIdata(test_meshXY_file)
    XY = mesh_XY['meshXY']
    test_xy_data = np.transpose(XY, (1, 0))
    return test_xy_data


def get_randData2Laplace(dim=2, data_path=None):
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
    mat_file_name = 'dataMat2pLaplace/meshXY.mat'
    mat_data = loadMatlabIdata(mat_file_name)
    XY = mat_data['meshXY']
    XY_T = np.transpose(XY, (1, 0))
    print('shdshd')