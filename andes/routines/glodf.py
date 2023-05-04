# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:59:15 2023

@author: rvaug
"""

import andes
import pandas as pd
import numpy as np

class GLODF:
    """
    Computes the Global Line Outage Distribution Factors (GLODF) for a power system.

    Parameters
    system : The power system model.
    """
    def __init__(self, system):
        self.system = system
        self.Bs = self.get_branch_susceptance()
        self.Ar = self.get_reduced_incidence()
        self.Bn = self.get_reduced_nodal_susceptance()

    def get_branch_susceptance(self):
        """
        Returns the branch susceptance matrix.

        Returns
        -------
        numpy.ndarray : Bs
            The L x L diagonal matrix of branch susceptances.
        """
        return np.diag(1/self.system.Line.x.v)
    
    def get_reduced_incidence(self):
        """
        Returns the reduced incidence matrix.

        Returns
        -------
        numpy.ndarray : Ar
            The L x (N-s) reduced incidence matrix where 's' is the number of 
            slack buses.
        """
        num_line = self.system.Line.n
        num_bus = self.system.Bus.n
        
        A = np.zeros((num_line, num_bus))
        
        for l in range(num_line):
            A[l,self.system.Line.bus1.v[l]-1] = -1
            A[l,self.system.Line.bus2.v[l]-1] =  1
        
        # delete all slack rows as required
        Ar = np.delete(A, np.asarray(self.system.Slack.bus.v)-1, axis=1)
        return Ar
    
    def get_reduced_nodal_susceptance(self):
        """
        Returns the reduced nodal susceptance matrix.

        Returns
        -------
        numpy.ndarray : Bn
            The (N-s) x (N-s) reduced nodal susceptance matrix where 's' is the
            number of slack buses.
        """
        return -1.0 * np.dot(np.dot(self.Ar.T, self.Bs), self.Ar)
            
    def get_isf(self):
        """
        Returns the injection shift factor (ISF) matrix.
        
        Returns
        -------
        numpy.ndarray : isf
            The L x (N-s) injection shift factor matrix.
        """
        psi = np.dot(np.dot(self.Bs, self.Ar), np.linalg.inv(self.Bn))
        
        return psi
    
    def get_ptdf(self, change_lines):
        """
        Returns the power transfer distribution factors for the given lines.
        
        Returns
        -------
        numpy.ndarray : ptdf
            The L x (lines) matrix.
        """        
        change_lines = np.atleast_1d(change_lines)
        psi = self.get_isf()
        
        slack = np.array(self.system.Slack.bus.v)
        non_slack = np.delete(np.arange(self.system.Line.n), slack - 1)
        
        # minus ones to make zero indexed
        bfrom = np.asarray(self.system.Line.bus1.v)[change_lines - 1]
        bto   = np.asarray(self.system.Line.bus2.v)[change_lines - 1]
        
        bfrom_idx = np.zeros_like(bfrom)
        bto_idx   = np.zeros_like(bto)
        for i in range(np.size(change_lines)):
            bfrom_idx[i] = np.argwhere(non_slack == bfrom[i] - 1)
            bto_idx[i]   = np.argwhere(non_slack == bto[i]   - 1)
            
        phi = psi[:,bfrom_idx] - psi[:,bto_idx]
        return phi
    
    def lodf(self, change_line):
        """
        Returns the line outage distribution factors (LODFs) matrix for single
        line outage
        
        Returns
        -------
        numpy.ndarray : glodf
            The 1 x L injection shift factor matrix, where 'o' is the number of 
            outages.
        """
        phi = self.get_ptdf(change_line)
        
        sigma = phi / (1 - phi[change_line - 1])
        sigma[change_line - 1] = 0
        
        return sigma #[np.arange(sigma.size) != change_line - 1]
        
    def flow_after_lodf(self, change_line):
        """
        Returns the line flows after the line outage
        
        Returns
        -------
        numpy.ndarray : flow_after
            The length L vector of line flows after the outage
        """
        sigma = np.squeeze(self.lodf(change_line))
        
        flow_before = self.system.Line.a1.e
        
        flow_after = flow_before + flow_before[change_line-1] * sigma
        flow_after[change_line-1] = 0
        
        return flow_after
    
    def glodf(self, change_lines):
        """
        Returns the generalized line outage distribution factors (GLODFs) 
        matrix for the line outages in the change_lines list.
        
        Returns
        -------
        numpy.ndarray : glodf
            The o x L injection shift factor matrix, where 'o' is the number of 
            outages.
        """
        change_lines = np.atleast_1d(change_lines)
        
        phi = self.get_ptdf(change_lines)
        
        # right side of equation is all lines
        right_side = phi.T
        
        # left side is identity - Phi of change lines
        Phi = right_side[:,change_lines-1]
        left_side = (np.eye(np.shape(Phi)[0]) - Phi)
        
        xi = np.linalg.solve(left_side, right_side)
        
        return xi
    
    def flow_after_glodf(self, change_lines):
        """
        Returns the line flows after the line outages given in change_lines
        
        Returns
        -------
        numpy.ndarray : flow_after
            The length L vector of line flows after the outages
        """
        change_lines = np.atleast_1d(change_lines)
        
        flow_before = self.system.Line.a1.e
    
        xi = self.glodf(change_lines)
        
        #GLODFs times flow before
        delta_flow = xi.T @ flow_before[change_lines-1]
        flow_after = flow_before + delta_flow
        
        # ensure lines that are out have no flow
        flow_after[change_lines-1] = 0
        
        return flow_after
    
if __name__ == "__main__":
    """
    Example code to test the GLODF class
    """
    print("GLODF Test")
    # load system
    ss = andes.load(andes.get_case("ieee14/ieee14_linetrip.xlsx"))
    
    # solve system
    ss.PFlow.run()
    
    # create GLODF object
    g = GLODF(ss)

    # lines to be taken out
    change_lines = [5, 6, 12]
    
    lines_before = np.copy(ss.Line.a1.e)
    lines_after = g.flow_after_glodf(change_lines)
    
    np.set_printoptions(precision=5)
    # print("flow Before:\n" + str(ss.Line.a1.e))
    # print("flow after GLODF:\n" + str(lines_after))
    
    # turn off lines and resolve
    for i in range(np.size(change_lines)):
        ss.Line.u.v[change_lines[i]-1] = 0
    ss.PFlow.run()
    # print("flow Re-solved:\n" + str(ss.Line.a1.e))
    
    lineflows = {
                 "bus1": ss.Line.bus1.v,
                 "bus2": ss.Line.bus2.v,
                 "P1 before": lines_before,
                 "P1 GLODF": lines_after,
                 "P1 re-solved": ss.Line.a1.e,
                 "error": np.abs(lines_after - ss.Line.a1.e)
                 }
    
    df_lineflow = pd.DataFrame(lineflows, index=ss.Line.idx.v)
    
    print(df_lineflow)
    
    mask = ss.Line.a1.e != 0
    mape = np.mean(np.abs((ss.Line.a1.e[mask] - lines_after[mask]) / ss.Line.a1.e[mask])) * 100
    print("mean absolute percent error: {:.3f}%".format(mape))
    