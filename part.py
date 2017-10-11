# coding: utf-8
#
# Author: HZG
#
# Part of relation prediction using RESCAL

import sys
sys.path.append("..")

import h5py
import logging
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sktensor import rescal_als

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Example Kinships')

def predict_rescal_als(T):
    '''
     RESCAL computing.

     Parameter:
         T: train sparse tensor.

     Return:
         P: result dense tensor.
    '''
    A, R, _, _, _ = rescal_als(
        T, 100, init='similarity', conv=1e-3,
        lambda_A=10, lambda_R=10
    )
    n = A.shape[0]
    P = np.zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = np.dot(A, np.dot(R[k], A.T))
    _log.info("Dtype of P is %s" % str(P.dtype))
    print "-------"
    return P

def accuracy(P, test_list):
    test_acc = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(0, test_list.shape[0]):
        true_index = int(test_list[i][2])
        sub_index = int(test_list[i][0]) - 1
        obj_index = int(test_list[i][1]) - 1
        predict_index = np.argmax(P[sub_index, obj_index, :])
        test_list[i][3] = predict_index
        if true_index == predict_index:
            test_acc[true_index] = test_acc[true_index] + 1
    e = float(P.shape[0])
    test_acc = test_acc / e
    return test_acc, test_list

def main():
    # Set logging
    logging.basicConfig(level=logging.INFO)
    # Load matlab file
    tensor_mat_path = './data/PartTensor.mat'
    mat = h5py.File(tensor_mat_path)
    # Train tensor
    T = mat['TrainTensor'][:].T
    P = mat['TestTensorList'][:].T

    e, k = T.shape[0], T.shape[2]

    _log.info(T.shape)
    _log.info(P.shape)

    X = [lil_matrix(T[:, :, i]) for i in range(k)]

    result_tensor = predict_rescal_als(X)
    predict_acc, predict_res = accuracy(result_tensor, P)
    print predict_acc
    np.savetxt('./data/predict_result.txt', predict_res)

if __name__ == '__main__':
    main()