# coding: utf-8
#
# Author: HZG
#
# Part of relation prediction using ADMM

import sys
sys.path.append("..")

import h5py
import logging
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sktensor import admm_als

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Example Kinships') 

def predict_admm_als(T, S, rank, lmbda1, lmbda2, lmbda3, rho):
    A1, B1, R, _, _, _ = admm_als(
        T, S, rank, conv=1e-3,
        lambda_1=lmbda1, lambda_2=lmbda2, lambda_3 = lmbda3, rho = rho
    )
    n = A1.shape[0]
    P = np.zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = np.dot(A1, np.dot(R[k], B1.T))
    _log.info("Dtype of P is %s" % str(P.dtype))
    print "-------"
    return P

def accuracy(P, test_list):
    test_acc = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(0, test_list.shape[0]):
        true_index = int(test_list[i][2])
        sub_index = int(test_list[i][0]) - 1
        obj_index = int(test_list[i][1]) - 1
        predict_index = np.argmax(P[sub_index, obj_index, :]) + 1
        test_list[i][3] = predict_index
        if true_index == predict_index:
            test_acc[true_index - 1] = test_acc[true_index - 1] + 1
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
    S = mat['TestTensorList'][:].T
    P = mat['tensorSimilarity'][:].T

    e, k = T.shape[0], T.shape[2]
    _log.info(T.shape)
    _log.info(S.shape)
    X = [lil_matrix(T[:, :, i]) for i in range(k)]

    # Init parameters
    rank = 1
    lmbda1 = 0
    lmbda2 = 0
    lmbda3 = 0
    rho = 0.0001
    max_acc = np.array([0.0, 0.0, 0.0, 0.0])
    # Computing
    while rank < 20:
        while lmbda1 < 5:
            while lmbda2 < 5:
                while lmbda3 < 5:
                    while rho < 5:
                        result_tensor = predict_admm_als(X, P, rank, lmbda1, lmbda2, lmbda3, rho)
                        predict_acc, predict_res = accuracy(result_tensor, S)
                        if predict_acc.min() > max_acc.min():
                            max_acc = predict_acc
                        print "rank: %d; lambda_1: %d; lambda_2: %d; lambda_3: %d; rho: %f." %(rank, lmbda1, lmbda2, lmbda3, rho)
                        print "predict_accuracy: " + str(predict_acc)
                        print "max_accuracy:" + str(max_acc)
                        save_filename = './data/predict_result_' + str(rank) + '_' + str(lmbda1) + '_' + str(lmbda2) + '_' + str(lmbda3) + '_' + str(rho) + '.txt'
                        np.savetxt(save_filename, predict_res)
                        rank += 1
                        lmbda1 += 0.5
                        lmbda2 += 0.5
                        lmbda3 += 0.5
                        rho += 0.1

if __name__ == '__main__':
    main()
    