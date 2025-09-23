import numpy as np
import pandas as pd
import time
from scipy.special import jv, kv
import json
import os
import random

import _pickle as cPickle
import sys
import zlib
from dataclasses import dataclass, replace
from scipy.integrate import quad,fixed_quad, nquad

from numpy.polynomial.legendre import leggauss


def load(name):
	compressed = open(name,"rb").read()
	data = cPickle.loads(zlib.decompress(compressed))
	return data


def save(data, name):
	compressed = zlib.compress(cPickle.dumps(data))
	f = open(name,"wb")
	try:
		f.writelines(compressed)
	except:
		f.write(compressed)
		f.close()


def regulate_IR(values, r, params):
	if params[0] == 'gauss':
		values *= np.exp(-(r**2)*params[1])
	elif params[0] == 'skin':
		values *= 1.0/(1 + np.exp(params[1]*((r/params[2]) - 1)))
	elif params[0] == 'exp':
		values *= np.exp(-r*params[1])
	return values


def map_to_interval(ns, ws, a, b):
	mapped_nodes = 0.5 * (b + a) + 0.5 * (b - a) * ns
	mapped_weights = 0.5 * (b - a) * ws
	return mapped_nodes, mapped_weights


@dataclass
class Kinematics:
	s: float = 0.0
	Q: float = 0.0
	x: float = 0.0
	t: float = 0.0
	y: float = 0.0
	phi: float = 0.0



class PionSSA:
	def __init__(self, **options):

		# define optional parameters
		replica = options.get('replica', 1)
		self.lambdaIR = options.get('lambdaIR', 0.3)
		self.deta = options.get('deta', 0.05)
		self.IR_params = options.get('IR_reg', [None, 0.0])
		fit_type = options.get('fit_type', 'pp')
		self.nucleon = options.get('nucleon', 'p')

		# define physical constants
		self.alpha_em = 1/137.0
		self.Nc = 3.0
		self.Nf = 3.0
		if self.nucleon == 'p':
			self.Zusq = 4.0/9.0
			self.Zdsq = 1.0/9.0
		elif self.nucleon == 'n':
			self.Zusq = 1.0/9.0
			self.Zdsq = 4.0/9.0 
		else:
			raise ValueError('Error: do not recognize nucleon', self.nucleon)
		self.Zssq = 1.0/9.0
		self.Zfsq = self.Zusq + self.Zdsq + self.Zssq

		self.current_dir = os.path.dirname(os.path.abspath(__file__))

		self.load_dipoles()
		self.load_params(fit_type)
		self.set_params(replica)



	def load_dipoles(self):

		#-- load polarized dipole amplitudes
		deta_str = 'd'+str(self.deta)[2:]
		polar_indir = f'/dipoles/{deta_str}-rc-largeNcq/'
		amps = ['QNSu', 'QNSd', 'QNSs']

		self.basis_dipoles = {}

		for iamp in amps:
			self.basis_dipoles[iamp] = {}
			input_dipole = 0
			for j, jamp in enumerate(amps):

				self.basis_dipoles[iamp][jamp] = {}
				for jbasis in ['eta', 's10','1']:

					self.basis_dipoles[iamp][jamp][jbasis] = load(f'{self.current_dir}{polar_indir}pre-cook-{jamp}-{jbasis}-{iamp}.dat')

		# grid values for dipoles
		n_psteps = len(self.basis_dipoles['Qu']['Qu']['eta'])
		self.s10_values = np.arange(0.0, self.deta*(n_psteps), self.deta)



	def load_params(self, fit_type='dis'):

		params_file = f'/dipoles/replica_params_{fit_type}.csv'
		fdf = pd.read_csv(self.current_dir + params_file)
		header = ['nrep'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2'] for ib in ['eta', 's10', '1']]
		fdf = fdf.dropna(axis=1, how='all')
		fdf.columns = header
		self.params = fdf
		print('--> loaded params from', params_file)


	def set_params(self, nreplica=1):

		sdf = self.params[self.params['nrep'] == nreplica]
		assert len(sdf) == 1, 'Error: more than 1 replica selected...'
		print('--> loaded replica', nreplica)

		ic_params = {}
		amps = ['QNSu', 'QNSd', 'QNSs']
		for amp in amps:
			ic_params[amp] = {}
			for basis in ['eta','s10','1']:
				ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]

		self.pdipoles = {}
		for iamp in amps:
			input_dipole = 0
			for j, jamp in enumerate(amps):
				for jbasis in ['eta', 's10','1']:
					temp_dipole = self.basis_dipoles[iamp][jamp][jbasis]
					input_dipole += ic_params[jamp][jbasis]*temp_dipole

			self.pdipoles[iamp] = input_dipole


	def get_dipoles(self):
		return self.pdipoles


	# F1 form factor of the proton
	def f1(self, t):
		pass


	# returns numerator of SSA in pb or fb (differential in Q^2, y, \phi, t)
	def get_ssa_numerator(self, kins, harmonic='sin(phi)'):

		Q,x,y,t,phi = kins.Q, kins.x, kins.y, kins.t, kins.phi
		W2 = (Q**2)/x

		if harmonic == 'sin(phi)':

			prefactor = -128*4*np.sqrt(2)*(self.Nc**2)*(self.alpha_em**2)*(Q**2)
			prefactor *= self.f1(t)*(1/np.sqrt(t))*(1/W2)






if __name__ == '__main__':

	test_kins = Kinematics()
	test_kins.x = 0.01
	test_kins.Q = 5
	test_kins.z = 0.4
	test_kins.s = 95**2
	test_kins.delta = 0.2
	test_kins.phi_Dp = 0
	test_kins.phi_kp = 0
	test_kins.pT = 10.0
	test_kins.y = (test_kins.Q**2)/(test_kins.s*test_kins.x)

	print(test_kins.y)
	print('pT:', test_kins.pT)
	print('root s', np.sqrt(test_kins.s))

	space = {
		'y' : [0.05, 0.95],
		'z' : [0.2, 0.5],
		'Q2' : [16, 100],
		# 't' : [0.01, 0.04],
		't' : 0.04,
		'phi_Dp' : [0, 2*np.pi],
		'phi_kp' : [0, 2*np.pi]
	}

	dj = DIJET(1, constrained_moments=True)
	dj.load_params('replica_params_pp.csv')
	dj.set_params(4)

	test_den = dj.get_integrated_xsec([test_kins.pT], test_kins.s, space, weight='1', points=7, kind='den')
	test_num = dj.get_integrated_xsec([test_kins.pT], test_kins.s, space, weight='1', points=7, kind='num')

	print(test_num, test_den, test_num/test_den)
	







