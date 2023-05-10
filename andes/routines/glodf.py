"""
Generalized Line Outage Distribution Factors (GLODF)

The GLODFs are useful for analyzing the impact of the failure of multiple
transmission lines on the power flow of the entire system. The GLODFs are a set
of coefficients that quantify the proportional change in power flows on all
other lines due to a change in power flow on a specific line. They are 
computed using the power transfer distribution factors (PTDFs), which describe 
the impact of a change in power flow on a specific generator or load on the 
power flows of all other generators or loads.
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
        Ar = np.delete(A, np.asarray(self.system.Bus.idx2uid(self.system.Slack.bus.v)), axis=1)
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
        
        Parameters
        ----------
        change_lines : line indeces
            List of line indices for which the GLODFs are to be computed.
        
        Returns
        -------
        numpy.ndarray : ptdf
            The L x (lines) matrix.
        """        
        change_lines = np.atleast_1d(change_lines)
        psi = self.get_isf()
        
        slack = np.array(self.system.Bus.idx2uid(self.system.Slack.bus.v))
        non_slack = np.delete(np.arange(self.system.Line.n), slack)
        
        bfrom = self.system.Bus.idx2uid(self.system.Line.get('bus1', change_lines))
        bto   = self.system.Bus.idx2uid(self.system.Line.get('bus2', change_lines))
        
        bfrom_idx = np.zeros_like(bfrom)
        bto_idx   = np.zeros_like(bto)
        
        for i in range(np.size(change_lines)):
            if (bfrom[i] in slack):
                bfrom_idx[i] = -1
            else:
                bfrom_idx[i] = np.argwhere(non_slack == bfrom[i])
            
            if (bto[i] in slack):
                bto_idx[i] = -1
            else:
                bto_idx[i]   = np.argwhere(non_slack == bto[i])
        
        # zeros row is needed because power injection at slack bus should yield psi = 0
        zeros_row = np.zeros((psi.shape[0], 1))
        psi_zero = np.hstack((psi, zeros_row))
        phi = psi_zero[:,bfrom_idx] - psi_zero[:,bto_idx]
        return phi
    
    def lodf(self, change_line):
        """
        Returns the line outage distribution factors (LODFs) matrix for single
        line outage
        
        Parameters
        ----------
        change_line : line index
            line indix for which the LODFs are to be computed.
        
        Returns
        -------
        numpy.ndarray : glodf
            The 1 x L injection shift factor matrix, where 'o' is the number of 
            outages.
        """
        phi = self.get_ptdf(change_line)
        
        uid_lines = self.system.Line.idx2uid(change_line)
        
        sigma = phi / (1 - phi[uid_lines])
        sigma[uid_lines] = 0
        
        return sigma #[np.arange(sigma.size) != change_line - 1]
        
    def flow_after_lodf(self, change_line):
        """
        Returns the line flows after the line outage
        
        Parameters
        ----------
        change_line : line index
            line indix for which the LODFs are to be computed.
        
        Returns
        -------
        numpy.ndarray : flow_after
            The length L vector of line flows after the outage
        """
        sigma = np.squeeze(self.lodf(change_line))
        
        uid_lines = self.system.Line.idx2uid(change_line)
        
        flow_before = self.system.Line.a1.e
        
        flow_after = flow_before + flow_before[uid_lines] * sigma
        flow_after[uid_lines] = 0
        
        return flow_after
    
    def glodf(self, change_lines):
        """
        Returns the generalized line outage distribution factors (GLODFs) 
        matrix for the line outages in the change_lines list.
        
        Parameters
        ----------
        change_lines : line indices
            List of line indices for which the GLODFs are to be computed.
            
        Returns
        -------
        numpy.ndarray : glodf
            The o x L injection shift factor matrix, where 'o' is the number of 
            outages.
        """
        uid_lines = np.atleast_1d(self.system.Line.idx2uid(change_lines))
        
        phi = self.get_ptdf(change_lines)
        
        # right side of equation is all lines
        right_side = phi.T
        
        # left side is identity - Phi of change lines
        Phi = right_side[:,uid_lines]
        left_side = (np.eye(np.shape(Phi)[0]) - Phi)
        
        xi = np.linalg.solve(left_side, right_side)
        
        return xi
    
    def flow_after_glodf(self, change_lines):
        """
        Returns the line flows after the line outages given in change_lines
        
        Parameters
        ----------
        change_lines : line indices
            List of line indices for which the GLODFs are to be computed.
        
        Returns
        -------
        numpy.ndarray : flow_after
            The length L vector of line flows after the outages
        """
        xi = self.glodf(change_lines)
        
        uid_lines = np.atleast_1d(self.system.Line.idx2uid(change_lines))
        
        flow_before = self.system.Line.a1.e
    
        
        #GLODFs times flow before
        delta_flow = xi.T @ flow_before[uid_lines]
        flow_after = flow_before + delta_flow
        
        # ensure lines that are out have no flow
        flow_after[uid_lines] = 0
        
        return flow_after
    