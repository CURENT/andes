from collections import defaultdict
from andes.shared import jac_names, jac_types, jac_full_names


class JacTriplet(object):
    """
    Storage class for Jacobian triplet lists.
    """
    def __init__(self):
        self.ijac = defaultdict(list)
        self.jjac = defaultdict(list)
        self.vjac = defaultdict(list)

    def clear_ijv(self):
        """
        Clear stored triplets for all sparse Jacobian matrices
        """
        for j_full_name in jac_full_names:
            self.ijac[j_full_name] = list()
            self.jjac[j_full_name] = list()
            self.vjac[j_full_name] = list()

    def append_ijv(self, j_full_name, ii, jj, vv):
        """
        Append triplets to the given sparse matrix triplets.

        Parameters
        ----------
        j_full_name : str
            Full name of the sparse Jacobian. If is a constant Jacobian, append 'c' to the Jacobian name.
        ii : array-like
            Row indices
        jj : array-like
            Column indices
        vv : array-like
            Value indices
        """
        if len(ii) == 0 and len(jj) == 0:
            return
        self.ijac[j_full_name].append(ii)
        self.jjac[j_full_name].append(jj)
        self.vjac[j_full_name].append(vv)

    def ijv(self, j_full_name):
        """
        Return triplet lists in a tuple in the order or (ii, jj, vv)
        """
        return self.ijac[j_full_name], self.jjac[j_full_name], self.vjac[j_full_name]

    def zip_ijv(self, j_full_name):
        """
        Return a zip iterator in the order of (ii, jj, vv)
        """
        return zip(*self.ijv(j_full_name))

    def merge(self, triplet):
        """
        Merge another triplet into this one.
        """
        for jname in jac_names:
            for jtype in jac_types:
                self.ijac[jname + jtype] += triplet.ijac[jname + jtype]
                self.jjac[jname + jtype] += triplet.jjac[jname + jtype]
                self.vjac[jname + jtype] += triplet.vjac[jname + jtype]
