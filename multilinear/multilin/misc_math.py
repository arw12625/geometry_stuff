import numpy as np


def lower_block_toeplitz_matrix(blocks, offset):
    if any(block.shape != blocks[0].shape or len(block.shape) != 2 for block in blocks):
        raise ValueError("All blocks must be matrices with the same shape")
    if offset < 0:
        raise NotImplementedError("Negative offsets are not supported yet.")

    # Handle zero width case (np.block does not support empty blocks)
    if len(blocks) == offset:
        return np.zeros((blocks[0].shape[0] * offset, 0))

    zero_block = np.zeros(blocks[0].shape)
    block_structure = [[blocks[i-j] if j <= i - offset else zero_block for j in range(len(blocks) - offset)] for i in range(len(blocks))]
    mat = np.block(block_structure)
    return mat
