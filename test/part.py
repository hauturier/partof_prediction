# coding: utf-8
#
# Author: HZG
#
# Part of relation prediction using RESCAL

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Example Kinships')

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
    
    result_tensor = predict_rescal_als(T)
    
    )
    
    X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]
    
    pass

if __name__ == '__main__':
    main()