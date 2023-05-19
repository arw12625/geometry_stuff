from multilin import multilinear

import numpy as np

class MultilinearSet:

    def __init__(self, multi_leq, out_var, multi_eq=None):
        self.multi_leq = multi_leq
        self.out_var = out_var
        self.multi_eq = multi_eq

        assert self.is_valid()

    def is_valid(self):
        if self.multi_leq is None:
            return False
        if self.out_var is None:
            return False
        if self.out_var not in self.multi_leq.axis_names:
            return False
        out_axis = self.multi_leq.get_axis_from_name(self.out_var)
        if self.multi_leq.homogenous[out_axis]:
                # or not all(h for ax, h in enumerate(self.multi_leq.homogenous) if ax != out_axis):
            return False
        if self.multi_eq is not None:
            if not self.multi_leq.has_same_names_ranges(self.multi_eq):
                return False

    @property
    def dimension(self):
        return self.multi_leq.get_axis_from_name(self.out_var) - 1

    @property
    def order(self):
        return self.multi_leq.get_order() - 1

    def is_polytope(self):
        return self.order == 1

    def set_containing_polytope(self, polytope, aux_var_name='T'):
        if not polytope.is_polytope():
            raise ValueError("Provided set is not a polytope")
        domain_name = (set(polytope.axis_names) - set(polytope.out_var)).pop()
        if domain_name not in self.multi_leq.axis_names:
            raise ValueError("Domain variable name missing")
        if not self.multi_leq.dim_range_names[domain_name] == polytope.multileq.dim_range_names[domain_name]:
            raise ValueError("Mismatch in domain variables")
        if self.multi_eq is not None or polytope.multi_eq is not None:
            raise NotImplementedError

        T_size = (self.multi_leq.get_dim_from_name(self.out_var), polytope.multileq.get_dim_from_name(polytope.out_var))
        T_mult_map = multilinear.multilinear_2D_linear(T_size, mat_name=aux_var_name, in_name=domain_name, out_name=self.out_var)
        c_multi = T_mult_map.compose(polytope.multileq)
        c_multi = c_multi.direct_sum(self, other, axis=None, out=None, new_axis_name=None, range_prefixes=('', ''), homogenous_action='keep_neither', homogenize_if_necessary=True):


        c_multi_eq = c_multi.get_component_as_multi(tuple(self.multi_leq.axis_names))
        c_multi_leq = c_multi.get_component_as_multi(tuple([]))

        return AlgSet(c_multi_leq, self.out_var, c_multi_eq)

    def copy(self):
        return MultilinearSet(self.multi_leq.copy(), self.out_var, self.multi_eq.copy() if self.multi_eq is not None else None)

    def robustify_vars(self, var_names, out_prefix='rob_'):
        # Need to derive constraint of the form {x | \forall w, M_eq(x,w) = 0 and M_leq(x,w) <= 0}
        if var_names is None:
            raise ValueError("Variables names cannot be None")
        if not var_names:
            return self.copy()
        if any(var_name == self.out_var for var_name in var_names):
            raise ValueError("Cannot robustify the output variable")
        if any(var_name not in self.multi_leq.axis_names for var_name in var_names):
            raise ValueError("Variable not present")

        out_axis = self.multi_leq.get_axis_from_name(self.out_var)
        var_out_axes = [self.multi_leq.get_axis_from_name(var_name) for var_name in var_names] + [out_axis]
        remaining_axes = [ax for ax, _ in enumerate(self.multi_leq.axis_names) if ax not in var_out_axes]
        var_first_permutation = var_out_axes + remaining_axes
        new_out_var_name = out_prefix + self.out_var
        new_axis_names = [new_out_var_name] +\
                         [self.multi_leq.axis_names[ax] for ax in remaining_axes]
        # TODO preserve original var names
        new_dim_names = {var_name: data for var_name, data in self.multi_leq.dim_range_names if var_name not in var_name and var_name != self.out_var}
        new_homogenous = [False] + [h for ax, h in self.multi_leq.homogenous if ax not in var_out_axes]

        robust_shape_leq = np.prod([self.multi_leq.tensor.shape[var_ax] for var_ax in var_out_axes])
        leq_reorder = np.transpose(self.multi_leq, var_first_permutation)
        leq_reorder.reshape(robust_shape_leq)
        leq_map = multilinear.Multilinear(new_axis_names, leq_reorder, new_dim_names, new_homogenous)

        eq_map = None
        if self.multi_leq is not None:
            robust_shape_eq = np.prod([self.multi_eq.tensor.shape[var_ax] for var_ax in var_out_axes])
            eq_reorder = np.transpose(self.multi_eq, var_first_permutation)
            eq_reorder.reshape(robust_shape_eq)
            eq_map = multilinear.Multilinear(new_axis_names, eq_reorder, new_dim_names, new_homogenous)

        robust_set = MultilinearSet(leq_map, new_out_var_name, eq_map)
        return robust_set
