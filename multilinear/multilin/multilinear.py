import numpy as np

class Multilinear:
    '''
    Class representing a multilinear map or equivalently a tensor.
    The axes of the tensor and ranges of their dimensions are named.
    Affine maps can be represented with homogenous coordinates

    # ex)
    m = Multilinear(['w', 'K', 'xu'], <numpy tensor (n, nm, n+m)>, dim_range_names={'xu':[('x',0,n),('u',n,m)]}, homogenous=['w'])
    '''
    def __init__(self, axis_names, tensor, dim_range_names=None, homogenous=None):
        self.axis_names = axis_names
        self.tensor = tensor

        # complete provided names for dimension ranges if incomplete
        if dim_range_names is None:
            dim_range_names = {}
        self.dim_range_names = {name: [(name, 0, tensor.shape[i])] for i, name in enumerate(axis_names)}
        self.dim_range_names.update(dim_range_names)

        if homogenous is None:
            homogenous = [False] * len(axis_names)
        self.homogenous = homogenous

        # Validate
        #assert self.is_valid()

    def __str__(self):
        return '\n'.join(['', str(self.axis_names), str(self.tensor), ''])

    def copy_names_ranges(self):
        '''
        Create a new multilinear map with the same names and ranges as the current one with a new tensor
        :return:
        '''
        return Multilinear(self.axis_names, np.ndarray(self.tensor.shape),
                           dim_range_names=self.dim_range_names, homogenous=self.homogenous)

    @property
    def order(self):
        return len(self.axis_names)

    def is_valid(self):
        # TODO: need to allow empty ranges
        if len(self.axis_names) != len(set(self.axis_names)):
            return False
        # dimension names should correspond to the orders in the tensor
        if len(self.axis_names) != len(self.tensor.shape):
            return False
        if set(self.axis_names) != self.dim_range_names.keys():
            return False
        # ensure range names are unique across dimensions
        if sum(len(ranges) for name, ranges in self.dim_range_names.items()) != \
               len({rname for name, ranges in self.dim_range_names.items() for rname, _, _ in ranges}):
            return False
        try:
            for i, name in enumerate(self.axis_names):
                ranges = self.dim_range_names[name]
                # ensure ranges are complete and correspond to tensor dimension
                start = 0
                for j in range(len(ranges)):
                    start = [rend for rname, rstart, rend in ranges if rstart == start and (rend > start)][0]
                if start != self.tensor.shape[i]:
                    return False
        except IndexError:
            return False

        if len(self.homogenous) != len(self.axis_names):
            return False

        return True

    def get_axis_from_name(self, name):
        return self.axis_names.index(name)

    def get_dim_from_name(self, name):
        return self.tensor.shape[self.get_axis_from_name(name)]

    def rename_dim(self, axis, new_name, new_range=None):
        self.axis_names[axis] = new_name
        if new_range is not None:
            self.dim_range_names = new_range
            if not self.is_valid():
                raise ValueError

    def rename_name(self, name, new_name, new_range_names=None):
        self.rename_dim(self.get_axis_from_name(name), new_name, new_range_names)

    def has_same_names_ranges(self, other, ignore_axes=None):
        '''
        Check if the provided multilinear map has the same names and ranges
        '''
        if len(self.axis_names) != len(other.axis_names):
            return False
        for i, name in enumerate(self.axis_names):
            if i in ignore_axes:
                continue
            name = self.axis_names[i]
            if name != other.axis_names[i]:
                return False
            self_ranges = set(self.dim_range_names[name])
            other_ranges = set(other.dim_range_names[name])
            if self_ranges != other_ranges:
                return False
        return True

    def scale(self, k, out=None):
        if out is None:
            out = self.copy_names_ranges()
        self.tensor.multiply(k, out=out)
        return out

    def add(self, other, out=None):
        '''
        Add the multilinear maps. Must have the same names and ranges
        '''
        if not self.has_same_names_ranges(other):
            raise ValueError("Multilinear has different names or ranges")
        if out is None:
            out = self.copy_names_ranges()
        out.tensor = self.tensor + other.tensor
        return out

    def direct_sum(self, other, axis=None, out=None, new_axis_name=None, range_prefixes=('', ''), homogenous_action='keep_neither', homogenize_if_necessary=True):
        make_first_homogenous = [ax for ax, h in enumerate(self.homogenous) if other.homogenous[ax] and not h]
        make_second_homogenous = [ax for ax, h in enumerate(other.homogenous) if self.homogenous[ax] and not h]
        first = self
        if make_first_homogenous or make_second_homogenous:
            if not homogenize_if_necessary:
                raise ValueError("Homogenous axis mismatch")

            if make_second_homogenous:
                first = self.copy()
                first.make_axes_homogenous([self.axis_names[ax] for ax in make_first_homogenous])
            if make_second_homogenous:
                other = other.copy()
                other.make_axes_homogenous([other.axis_names[ax] for ax in make_second_homogenous])

        if axis is None:
            diff = set(first.axis_names) - set(other.axis_names)
            if len(diff) != 1:
                raise ValueError("Could not infer axis to sum")
            axis = first.get_axis_from_name(diff.pop())

        if not first.has_same_names_ranges(other, ignore_axes=[axis]):
            raise ValueError("Multilinear has different names or ranges")
        if homogenous_action not in ['keep_neither', 'keep_first', 'keep_second', 'keep_both', 'add']:
            raise ValueError("Unrecognized homgenous action")

        remove_first_homogenous = first.homogenous[axis] and homogenous_action in ['keep_neither', 'keep_second', 'add']
        remove_second_homogenous = other.homogenous[axis] and homogenous_action in ['keep_neither', 'keep_first']
        if first.homogenous[axis] and not remove_first_homogenous:
            raise Warning("Behavior for homogenous dimensions not at the end of the range is undefined.")

        if out is None:
            axis_names = first.axis_names.copy()
            if new_axis_name is None:
                new_axis_name = first.axis_names[axis] + "+" + other.axis_names[axis]
            axis_names[axis] = new_axis_name

            shape_list = list(first.tensor.shape)
            shape_list[axis] += other.tensor.shape[axis]

            dim_range_names = {}
            dim_range_names.update(first.dim_range_names)
            dim_range_names.pop(first.axis_names[axis])
            self_axis_dim_range = [(range_prefixes[0] + rname, rstart, rend) for rname, rstart, rend in first.dim_range_names[first.axis_names[axis]]]
            if remove_first_homogenous:
                self_axis_dim_range.pop()
                shape_list[axis] -= 1
            other_axis_dim_range = [(range_prefixes[1] + rname, rstart, rend) for rname, rstart, rend in other.dim_range_names[other.axis_names[axis]]]
            if remove_second_homogenous:
                other_axis_dim_range.pop()
                shape_list[axis] -= 1
            dim_range_names[new_axis_name] = self_axis_dim_range + other_axis_dim_range

            homogenous = first.homogenous
            homogenous[axis] = (first.homogenous[axis] and not remove_first_homogenous) or (other.homogenous[axis] and not remove_second_homogenous)

            shape = tuple(shape_list)
            out = Multilinear(axis_names, np.ndarray(shape), dim_range_names, homogenous)

        keep_slc = slice(None)
        remove_slc = [slice(0, -1) if ax == axis else slice(None) for ax in range(first.order)]
        if homogenous_action == 'add':
            hom_slc = [slice(-1, None) if ax == axis else slice(None) for ax in range(first.order)]
            np.concatenate([first.tensor[remove_slc], other.tensor[remove_slc],
                            first.tensor[hom_slc] + other.tensor[hom_slc]], axis=axis, out=out.tensor)
        else:
            first_slc = remove_slc if remove_first_homogenous else keep_slc
            second_slc = remove_slc if remove_second_homogenous else keep_slc
            np.concatenate([first.tensor[first_slc], other.tensor[second_slc]], axis=axis, out=out.tensor)

        return out

    def compose(self, other, out=None, range_prefixes=('', ''), homogenize_if_necessary=True):
        # Todo add support for range subs
        contract_axes = list(zip(*[(ax1, ax2) for ax1, n1 in enumerate(self.axis_names)
                                           for ax2, n2 in enumerate(other.axis_names)
                                           if n1 == n2]))
        # TODO: add support for homogenous
        if any(other.homogenous[ax_pair[1]] and not self.homogenous[ax_pair[0]] for ax_pair in zip(*contract_axes)):
            raise ValueError("Homogenous axes do not match")
        make_homogenous = [ax_pair[0] for ax_pair in zip(*contract_axes) if
                           self.homogenous[ax_pair[0]] and not other.homogenous[ax_pair[1]]]
        if make_homogenous:
            if not homogenize_if_necessary:
                raise ValueError("Homogenous axes do not match")

            # TODO: make behavior explicit
            # When all of other's axes need to be homogenized, we set the corner to one for convenience
            # This is useful when using a homogenous multilinear map to represent an affine map and applying that map to a nonhomogenous vector
            component_map = {(): np.array(1)} if len(make_homogenous) == other.order() else None
            other = other.copy()
            other.make_axes_homogenous([other.axis_names[ax] for ax in make_homogenous],
                                       component_map=component_map)

        #if any(self.homogenous[ax] for ax in contract_axes[0]) or any(other.homogenous[ax] for ax in contract_axes[1]):
        #    raise NotImplementedError("Composition for homogenous axes is not supported yet.")
        self_uncommon_axes = [ax for ax in range(len(self.axis_names)) if ax not in contract_axes[0]]
        other_uncommon_axes = [ax for ax in range(len(other.axis_names)) if ax not in contract_axes[1]]
        # TODO: compare with np.einsum
        result = np.tensordot(self.tensor, other.tensor, axes=contract_axes)

        if out is None:
            axis_names = [self.axis_names[ax1] for ax1 in self_uncommon_axes] + \
                         [other.axis_names[ax1] for ax1 in other_uncommon_axes]
            dir_range_names = {self.axis_names[ax]: [(range_prefixes[0]+rname, rstart, rend) for rname, rstart, rend in self.dim_range_names[self.axis_names[ax]]]
                               for ax in self_uncommon_axes} |\
                              {other.axis_names[ax]: [(range_prefixes[1]+rname, rstart, rend) for rname, rstart, rend in other.dim_range_names[other.axis_names[ax]]]
                               for ax in other_uncommon_axes}
            homogenous = [self.homogenous[ax] for ax in self_uncommon_axes] +\
                         [other.homogenous[ax] for ax in other_uncommon_axes]

            out = Multilinear(axis_names, result, dir_range_names, homogenous)
        else:
            out.tensor = result
        return out


    def set_homogenous_components(self, component_map):

        axis_names_set = set(self.axis_names)
        for names in component_map.keys():
            diff = axis_names_set - set(names)
            if not all(self.homogenous[self.get_axis_from_name(ax_name)] for ax_name in diff):
                raise ValueError("Cannot add component corresponding to non-homogenous axes")

        for axis_names, data in component_map.items():
            axes = [self.get_axis_from_name(ax_name) for ax_name in axis_names]
            partial_permutation = sorted(range(len(axis_names)), key=lambda ind: axes[ind])

            slc = [(slice(0, -1) if self.homogenous[ax] else slice(None)) if ax in axes else slice(-1, None) for ax in range(self.order)]
            data_slc = [slice(None) if ax in axes else np.newaxis for ax in range(self.order)]
            self.tensor[slc] = np.transpose(data, partial_permutation)[data_slc]

    def make_axes_homogenous(self, axes_to_homogenize=None, component_map=None, range_prefix='hom_'):
        if not axes_to_homogenize and axes_to_homogenize is not None:
            return self

        if axes_to_homogenize is None:
            axes_to_homogenize = set()
            axis_names_set = set(self.axis_names)
            for names in component_map.keys():
                axes_to_homogenize.update(axis_names_set - set(names))
            axes_to_homogenize = list(axes_to_homogenize)

        if any([self.homogenous[self.get_axis_from_name(ax_name)] for ax_name in axes_to_homogenize]):
            raise ValueError("Cannot homogenize homogeneous axis.")

        for ax_name in axes_to_homogenize:
            ax = self.get_axis_from_name(ax_name)
            self.dim_range_names[ax_name].append((range_prefix + ax_name, self.tensor.shape[ax], self.tensor.shape[ax]+1))
            self.homogenous[ax] = True

        self.tensor = np.pad(self.tensor, [(0, 1) if ax_name in axes_to_homogenize else (0, 0)
                                           for ax_name in self.axis_names])

        if component_map:
            self.set_homogenous_components(component_map)


        return self

    def make_homogenous(self, component_map=None, range_prefix='hom_', set_corner=True):

        if set_corner:
            if component_map is None:
                component_map = {}
            component_map[()] = np.array(1)
        return self.make_axes_homogenous(self.axis_names, component_map, range_prefix)

    # Todo do this the right way
    def copy(self):
        return Multilinear(self.axis_names.copy(), self.tensor.copy(), self.dim_range_names.copy(), self.homogenous.copy())

    def is_consistent(self, other):
        ignore = list(set(self.axis_names).symmetric_difference(set(other.axis_names)))
        return self.has_same_names_ranges(other, ignore_axes=ignore)


def multilinear_from_homogenous_components(axis_names, dim_range_names=None, axes_to_homogenize=None, component_map=None, base_tensor=None, set_corner=True):

    if axes_to_homogenize is None:
        axes_to_homogenize = axis_names

    axis_names_set = set(axis_names)
    if base_tensor is None:
        if component_map is not None:
            for comp_name, comp_tensor in component_map.items():
                if axis_names_set == set(comp_name):
                    # TODO need to permute
                    raise NotImplementedError("NEED TO HANDLE PERMUTATION")
                    base_tensor = comp_tensor
        if base_tensor is None:
            raise ValueError("Base tensor to homogenize not provided")

    #homogenous = [(name in axes_to_homogenize) for name in axis_names]

    multi_map = Multilinear(axis_names, base_tensor, dim_range_names)
    if axis_names_set == set(axes_to_homogenize):
        multi_map.make_homogenous(component_map, set_corner=set_corner)
    else:
        multi_map.make_axes_homogenous(axes_to_homogenize, component_map)

    return multi_map


def matrix_mult_tensor(mat_shape):
    if len(mat_shape) != 2:
        raise ValueError("Shape must be a 2-tuple corresponding to a matrix")
    size = mat_shape[0] * mat_shape[1]
    tensor = np.zeros((mat_shape[1], size, mat_shape[0]))
    one_inds = list(zip(*[(k, i*mat_shape[1]+k, i) for i in range(mat_shape[0]) for k in range(mat_shape[1])]))
    if len(one_inds) > 0:
        tensor[one_inds[0], one_inds[1], one_inds[2]] = 1

def multilinear_2D_linear(mat_shape, mat_name, in_name, out_name):
    raise NotImplementedError
    return None

def multilinear_2D_affine(mat_shape, mat_name, in_name, out_name):
    # represent an 2-D affine map <out = L in + r> with a homogenous tensor over out, [L,r], in
    # The basis for [L,r] is the standard row-major basis for the matrix L followed by the standard basis for the vector r
    if len(mat_shape) != 2:
        raise ValueError("Shape must be a 2-tuple corresponding to a matrix")
    # The tensor is composed of two parts, one representing the multiplication by L and one representing adding the constant vector r
    mult_size = mat_shape[0] * mat_shape[1]
    const_size = mat_shape[0]
    # Construct the multiplication part of the tensor
    mult_tensor = np.zeros((mat_shape[0], mult_size+const_size, mat_shape[1]))
    mult_one_inds = list(zip(*[(i, i*mat_shape[1]+k, k) for i in range(mat_shape[0]) for k in range(mat_shape[1])]))
    if len(mult_one_inds) > 0:
        mult_tensor[mult_one_inds[0], mult_one_inds[1], mult_one_inds[2]] = 1
    # Construct the multilinear map from the multiplication tensor
    axis_names = [out_name, mat_name, in_name]
    dim_range_names = {mat_name: [(mat_name+"_mult", 0, mult_size),
                                   (mat_name+"_const", mult_size, mult_size + const_size)]}
    '''
    out = Multilinear(axis_names, mult_tensor, dim_range_names=
                      {mat_name: [(mat_name+"_mult", 0, mult_size),
                                   (mat_name+"_const", mult_size, mult_size + const_size)]})
    '''
    # Add the constant part of the tensor as a homogenous component
    const_tensor = np.vstack([np.zeros((mult_size, const_size)), np.eye(const_size)])
    #out.make_homogenous(component_map=)

    out = multilinear_from_homogenous_components(axis_names, dim_range_names=dim_range_names,
                                                 axes_to_homogenize=[in_name], component_map={(mat_name, out_name): const_tensor},
                                                 base_tensor=mult_tensor)

    return out
