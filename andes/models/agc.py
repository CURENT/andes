
from cvxopt import matrix, spmatrix
from cvxopt import mul, div, exp
from andes.consts import *
from .base import ModelBase
from andes.utils.math import zeros
import logging

logger = logging.getLogger(__name__)


class BArea(ModelBase):
    def __init__(self, system, name):
        super(BArea, self).__init__(system, name)
        self._group = 'Calculation'
        self._data.update({'area': None,
						   'syn': None,
						   'beta': 0,
                           })
        self._mandatory.extend(['area','syn','beta'])
        self._algebs.extend(['Pexp','fcoi','ace'])
        self.calls.update({'gcall': True, 
						   'init1': True,
                           'jac0': True,
                           })
        self._service.extend(['P0','Mtot','M','usyn','wsyn'])
        self._fnamey.extend(['P_{exp}','f_{coi}','ace'])
        self._params.extend(['beta'])
        self._init()

    def init1(self, dae):

        for item in self._service:
            self.__dict__[item] = [[]] * self.n


		# Start with frequency
        for idx, item in enumerate(self.syn):
                self.M[idx] = self.read_data_ext('Synchronous', field='M', idx=item)
                self.Mtot[idx] = sum(self.M[idx])
                self.usyn[idx] = self.read_data_ext('Synchronous', field='u', idx=item)
                self.wsyn[idx] = self.read_data_ext('Synchronous', field='omega', idx=item)
                dae.y[self.fcoi[idx]] = sum(mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]

		#Get BA Export Power
        self.copy_data_ext('Area', field='area_P0', dest='P0', idx=self.area)
        dae.y[self.Pexp] = self.P0

        dae.y[self.ace] = 0

    def gcall(self, dae):
        P = self.read_data_ext('Area', field='area_P0', idx=self.area)
        dae.g[self.Pexp] = dae.y[self.Pexp] - P

        for idx, item in enumerate(self.syn):
                self.wsyn[idx] = self.read_data_ext('Synchronous', field='omega', idx=item)
                dae.g[self.fcoi[idx]] = dae.y[self.fcoi[idx]] - sum(mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]
        
        ACE = (self.P0-P) - mul(self.beta,(1 - dae.y[self.fcoi]))

        dae.g[self.ace] = dae.y[self.ace] - ACE


    def jac0(self, dae):
        dae.add_jac(Gy0, 1, self.Pexp, self.Pexp)
        dae.add_jac(Gy0, 1, self.fcoi, self.fcoi)
        dae.add_jac(Gy0, 1, self.ace, self.ace)
        dae.add_jac(Gy0, 1, self.ace, self.Pexp)
        dae.add_jac(Gy0, -self.beta, self.ace, self.fcoi)



# Needs to be updated with newest Andes version code #
#class AGC(ModelBase):
#	def __init__(self, system, name):
#		super(AGC, self).__init__(system, name)
#		self._group = 'Calculation'
#		self._data.update({'BA': None,
#						   'syn': None,
#						   'Ki': 0,
#						   'Td': 0,
 #                          })
	#	self._mandatory.extend(['BA','syn','Ki','Td'])
	#	self._states.extend(['Pagc'])
	#	self.calls.update({'gcall': True, 
	#					   'init1': True,
	#					   'jac0': True,
	#					   'fcall': True,
     #                      })
		#self._service.extend(['Mtot','M','usyn','ace','pm'])
	#	self._fnamex.extend(['P_{agc tot}'])
#		self._params.extend(['Ki','Td'])
#		self._init()

#	def init1(self, dae):
#
#		for item in self._service:
#			self.__dict__[item] = [[]] * self.n
#		
#		for idx, item in enumerate(self.syn):
#				self.M[idx] = self.read_data_ext('Synchronous', field='M', idx=item)
#				self.Mtot[idx] = sum(self.M[idx])
#				self.usyn[idx] = self.read_data_ext('Synchronous', field='u', idx=item)
#$				self.pm[idx] = self.read_data_ext('Synchronous', field='pm', idx=item)
#		self.copy_data_ext('BA', field='ace', idx=self.BA)
#
#	def gcall(self, dae):
#		for idx, item in enumerate(self.syn):
#			Kgen = div(self.M[idx],self.Mtot[idx])
#			Pgen = mul(Kgen,dae.x[self.Pagc[idx]])
#			dae.g[self.pm[idx]] =- Pgen


#	def fcall(self, dae):
#		dae.f[self.Pagc] = mul(div(self.Ki,self.Td  ),dae.y[self.ace]) - mul(div(1,self.Td), dae.x[self.Pagc])
#
#	def jac0(self, dae):
#		dae.add_jac(Fy0, div(self.Ki,self.Td), self.Pagc, self.ace)
#		dae.add_jac(Fx0, -div(1,self.Td), self.Pagc, self.Pagc)	
#
#		for idx, item in enumerate(self.syn):
#			Kgen = div(self.M[idx],self.Mtot[idx])
#			dae.add_jac(Gx, -Kgen, self.pm[idx], self.Pagc)	
#