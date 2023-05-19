import misc_math
from multilin import multilinear

import numpy as np

class LinSys:

    def __init__(self, Aseq, Bseq, state_name='x', input_name='u', dist_name='w'):
        if isinstance(Aseq, np.ndarray):
            Aseq = [Aseq]
            Bseq = [Bseq]
        self.Aseq = Aseq
        self.Bseq = Bseq
        self.state_name = state_name
        self.input_name = input_name
        self.dist_name = dist_name

        assert self.is_valid()

    def is_valid(self):
        if len(self.Aseq) != len(self.Bseq) or len(self.Aseq) == 0:
            return False
        if any(A.shape != (self.state_dim, self.state_dim) or
               B.shape != (self.state_dim, self.input_dim)
               for A, B in zip(self.Aseq, self.Bseq)):
            return False
        return True

    @property
    def state_dim(self):
        return self.Aseq[0].shape[0]

    @property
    def input_dim(self):
        return self.Bseq[0].shape[1]

    def closed_loop_map(self, horizon, controller_map):
        '''
        Construct a multilinear map from the disturbance sequence and the controller gains to the state and input sequences
        :param horizon:
        :param controller_map:
        :return:
        '''
        n = self.state_dim
        m = self.input_dim

        # State response to input (offset = 1 as input is not applied to initial state)
        u_x_map = multilinear.Multilinear([self.state_name, self.input_name],
                                          _system_matrix(horizon, self.Aseq, self.Bseq, offset=1))
        # State response to disturbance (offset = 0 as disturbance is applied to initial state)
        w_x_map = multilinear.Multilinear([self.state_name, self.dist_name],
                                          _system_matrix(horizon, self.Aseq, [np.eye(n)] * len(self.Aseq), offset=0))
        # State response to disturbance through controller
        w_x_map_u = u_x_map.compose(controller_map)
        # Complete state response to disturbance
        w_x_map_u.make_axes_homogenous(component_map={tuple(w_x_map.axis_names): w_x_map.tensor})

        closed_loop_map = w_x_map_u.direct_sum(controller_map)
        return closed_loop_map

    def affine_dist_feedback_controller(self, horizon, cont_name):
        Lshape = (horizon * self.input_dim, (horizon + 1) * self.state_dim)
        mult_map = multilinear.multilinear_2D_affine(Lshape, cont_name, self.dist_name, self.input_name)
        return mult_map


def _system_matrix(horizon, Aseq, Bseq, offset):

    # For time invariant sys, can compute system matrix as Toeplitz matrix more efficiently
    if len(Aseq) == 1:
        blocks = [Bseq[0]]
        for i in range(horizon):
            blocks.append(Aseq[0] @ blocks[-1])
        mat = misc_math.lower_block_toeplitz_matrix(blocks, offset)
        return mat

    n = Aseq[0].shape[0]
    m = Bseq[0].shape[1]
    height = horizon + 1
    width = horizon - offset + 1
    if width < 0:
        width = 0
    mat = np.zeros((n * height, m * width))
    if width <= 0:
        return mat

    for i in range(height):
        if i > offset:
            Ai = Aseq[i % len(Aseq)]
            mat[i*n:(i+1)*n][0:m*(i-offset)] = Ai @ mat[i*n:(i+1)*n][0:m*(i-offset)]
        if i >= offset:
            mat[i * n:(i + 1) * n][m * (i - offset):m * (i - offset + 1)] = Bseq[i % len(Aseq)]

    return mat