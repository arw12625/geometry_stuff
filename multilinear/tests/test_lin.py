import numpy as np

import multilinear
import lin_sys

def test_mult():
    K = np.array([[1,2],[3,4]])
    K_map = multilinear.Multilinear(['K'], K.flatten())
    x = np.array([1, -1])
    x_map = multilinear.Multilinear(['x'], x)
    mult_map = multilinear.multilinear_matrix_mult((K.shape[0], K.shape[1]), 'K', 'x', 'y')
    print(K_map)
    print(x_map)
    print(mult_map)

    tmp = mult_map.compose(K_map)
    print(tmp)
    tmp = tmp.compose(x_map)

    print(tmp)

    assert True


def test_lin_sys():
    n = 10
    m = 10

    A = np.eye(n)
    B = np.eye(n)
    lin = lin_sys.LinSys(A, B)

    horizon = 10
    controller_map = lin.affine_dist_feedback_controller(horizon, 'L')
    print(controller_map)
    closed_loop = lin.closed_loop_map(horizon, controller_map)
    print(closed_loop)

    w = np.concatenate([np.random.randn(n * (horizon)), np.zeros(n)], axis=0)
    wmap = multilinear.multilinear_from_homogenous_components(['w'], base_tensor=w)
    print(wmap)
    L = np.concatenate([np.zeros((n*m*(horizon+1)*horizon)), -w[0:n*horizon]], axis=0)
    Lmap = multilinear.multilinear_from_homogenous_components(['L'], base_tensor=L)
    print(Lmap)

    tmp = closed_loop.compose(wmap)
    tmp = tmp.compose(Lmap)
    print(tmp)

if __name__ == '__main__':
    test_lin_sys()
