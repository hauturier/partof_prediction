# coding: utf-8
#
# Author: HZG
#
# Part of relation prediction using RESCAL

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Example Kinships')
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sktensor import rescal_als

def predict_rescal_als(T):
    '''
     RESCAL computing.

     Parameter:
         T: train sparse tensor.

     Return:
         P: result dense tensor.
    '''
    A, R, _, _, _ = rescal_als(
        T, 100, init='nvecs', conv=1e-3,
        lambda_A=10, lambda_R=10
    )
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P

def accuracy(P, test_list):
    test_acc = np.array([0.0, 0.0, 0.0, 0.0])
    for row in test_list:
        true_index = row[2]
        predict_index = P[row[0],row[1],:].argmax()
        
        if true_index == predict_index:
            test_acc[true_index] = test_acc[true_index] + 1
    e = float(P.shape[0])
    test_acc = test_acc / e
    return 
    pass

def main():
    # Set logging
    logging.basicConfig(level=logging.INFO)
    
    # Load matlab file
    tensor_mat_path = './data/PartTensor.mat'
    mat = loadmat(tensor_mat_path)

    # Train tensor
    T = mat['TrainTensor']
    P = mat['TestTensorList']
    
    e,k = T.shape[0], T.shape[2]
    
    _log.info('Datasize: %d x %d x %d | No. of classes: %d' % (
        T[0].shape + (len(T),) + (k,))

    X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]
    result_tensor = predict_rescal_als(X)
    

    pass

if __name__ == '__main__':
    main()