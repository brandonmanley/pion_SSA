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


def map_to_interval(ns, ws, a, b):
	mapped_nodes = 0.5 * (b + a) + 0.5 * (b - a) * ns
	mapped_weights = 0.5 * (b - a) * ws
	return mapped_nodes, mapped_weights


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
		self.deta = options.get('deta', 0.02)
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


	#-- load polarized non-singlet dipole amplitudes
	def load_dipoles(self):

		deta_str = 'd'+str(self.deta)[2:]
		polar_indir = f'/dipoles/{deta_str}-rc/'
		amps = ['QNSu', 'QNSd', 'QNSs']

		self.basis_dipoles = {}
		for ibasis in ['eta', 's10', '1']:
			self.basis_dipoles[ibasis] = load(f'{self.current_dir}{polar_indir}pre-cook-{ibasis}.dat')

		# grid values for dipoles
		nsteps = len(self.basis_dipoles['eta'])
		self.s10_values = np.arange(0.0, self.deta*(nsteps), self.deta)


	#-- load initial condition parameters for non-singlet dipole amplitudes
	def load_params(self, fit_type='dis'):

		params_file = f'/dipoles/replica_params_{fit_type}.csv'
		fdf = pd.read_csv(self.current_dir + params_file)
		header = ['nrep'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'QNSu', 'QNSd', 'QNSs', 'GT', 'G2'] for ib in ['eta', 's10', '1']]
		fdf = fdf.dropna(axis=1, how='all')
		fdf.columns = header
		self.params = fdf
		print('--> loaded params from', params_file)


	#-- set initial condition parameters for non-singlet dipole amplitudes
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

		self.dipoles = {}
		for iamp in amps:
			input_dipole = 0
			for ibasis in ['eta', 's10','1']:
				input_dipole += ic_params[iamp][ibasis]*self.basis_dipoles[ibasis]
			self.dipoles[iamp] = input_dipole


	def get_dipoles(self):
		return self.dipoles
	
	def z(self, eta, W2):
		return ((self.lambdaIR**2)/W2)*np.exp(np.sqrt((2*np.pi)/self.Nc)*eta)

	def x12(self, s12):
		return (1/self.lambdaIR)*np.exp(-np.sqrt((2*np.pi)/self.Nc)*s12)

	#-- charge form factor of the proton
	def f1(self, t):
		Q02 = 0.71 # GeV^2
		return (1+(t/Q02))**(-2)

	#-- pion light-cone wavefunction (HZW model is from 1305.7391, see Table I)
	def psi(self, x12, z, model='HZW'):
		if model == 'HZW':
			anorm = 24.80
			beta = 0.589
			mq = 0.3
			return anorm * np.sqrt(np.pi * 8 * (beta**2) * z * (1-z)) * np.exp(- mq**2 / (8 * (beta**2) * z * (1-z))) * np.exp(-(x12**2) * (2*(beta**2) * z * (1-z)))
		else: 
			raise ValueError(f'model {model} not recognized')
	
	#-- photon-pion wavefunction overlaps
	def phi2(self, x12, z, Q):
		# print('z', z)
		# print('phi', x12*Q*np.sqrt(z*(1-z)))
		return np.sqrt(2*self.Nc) * (1/np.pi) * ((z*(1-z))**1.5) * Q * kv(1, x12*Q*np.sqrt(z*(1-z)))

	def phi1L(self, x12, z, Q):
		return  - np.sqrt(2*self.Nc) * (1/np.pi) * ((z*(1-z))**2) * Q * kv(0, x12*Q*np.sqrt(z*(1-z)))
	
	def phi1T(self, x12, z, Q):
		return np.sqrt(0.5*self.Nc) * (1/np.pi) * (z*(1-z)) * (1-2*z) * Q * kv(1, x12*Q*np.sqrt(z*(1-z)))

	#-- integrates function f over z: [0,1] and x_{12}:[0, \infty]
	def integrate(self, f, W2, s12_max=5, qcd=False):

		etaW2 = np.sqrt(self.Nc/(2*np.pi))*np.log(W2/(self.lambdaIR**2))
		n_eta = round(etaW2/self.deta)
		n_s12 = len(self.dipoles['QNSu'])
		s12_max = self.deta*n_s12
		eta = np.linspace(0, etaW2 - self.deta, n_eta)
		s12 = np.linspace(0, s12_max, n_s12)

		assert n_eta < n_s12, 'Error: going over the limit!'

		eta_grid, s12_grid = np.meshgrid(eta, s12, indexing="ij")
		jacobian = np.exp(np.sqrt((2*np.pi)/self.Nc) * (eta_grid - 0.5*s12_grid))

		F = f(eta_grid, s12_grid)

		if qcd:
			dipole_factor = self.Zusq * self.dipoles['QNSu'][:F.shape[0], :F.shape[1]]
			dipole_factor -= self.Zdsq * self.dipoles['QNSd'][:F.shape[0], :F.shape[1]]
			F *= dipole_factor
		
		# jacobian = np.exp(np.sqrt((2*np.pi)/self.Nc) * (eta[:, None] - s12[None, :]/2))
		# F = f(eta[:, None], s12[None, :])

		integral = np.sum(jacobian * F) * (self.deta**2)
		prefactor = (np.pi/self.Nc)*(self.lambdaIR/W2)
		return prefactor * integral
	


	#-- returns SSA in pb or fb (differential in Q^2, y, \phi, t)
	def get_ssa(self, kins, harmonic='sin(phi)'):

		Q,x,y,t,phi = kins.Q, kins.x, kins.y, kins.t, kins.phi
		W2 = (Q**2)/x

		num_prefactor = 0.5 * (self.alpha_em) / (8 * np.pi**2 * Q**2 * y)

		def qed_integrand(eta, s12):
			f = (1/(1-self.z(eta, W2))) * self.x12(s12)**2 
			f *= self.phi2(self.x12(s12), self.z(eta, W2), Q) 
			f *= self.psi(self.x12(s12), self.z(eta, W2))
			return f

		if harmonic == 'sin(phi)':
			def qcd_integrand(eta, s12):
				f = (1/(1-self.z(eta, W2))) * (1/self.z(eta, W2)**2) * self.x12(s12) 
				f *= self.phi1L(self.x12(s12), self.z(eta, W2), Q) 
				f *= self.psi(self.x12(s12), self.z(eta, W2))
				return f
			
			num = 8*np.sqrt(2) * (self.alpha_em**2) * (self.f1(t)/np.sqrt(t)) * (1/W2) 
			num *= self.integrate(qed_integrand, W2)
			num *= self.integrate(qcd_integrand, W2, qcd=True)

			num_prefactor *= (2-y)*np.sqrt(2-2*y) 

		elif harmonic == 'sin(2phi)':
			def qcd_integrand(eta, s12):
				f = (1/self.z(eta, W2)**2) * (self.x12(s12)**2)
				f *= self.phi1T(self.x12(s12), self.z(eta, W2), Q) 
				f *= self.psi(self.x12(s12), self.z(eta, W2))
				return f
			
			num = 4 * (self.alpha_em**2) * self.f1(t) * (1/W2) 
			num *= self.integrate(qed_integrand, W2)
			num *= self.integrate(qcd_integrand, W2, qcd=True)

			num_prefactor *= -(2-2*y)


		den = (self.alpha_em**4) * np.pi * (self.Zusq - self.Zdsq)**2 * ((1 + (1-y)**2)/(x*(Q**2) )) * ((self.f1(t)**2)/t)
		den *= self.integrate(qed_integrand, W2)**2

		ssa = {}
		ssa['num'] = num*num_prefactor 
		ssa['den'] = den
		ssa['ssa'] = ssa['num']/ssa['den']

		# convert to fb 
		ssa['num'] *= 0.3894*(10**12)
		ssa['den'] *= 0.3894*(10**12)
		return ssa



	#-- returns SSA in pb or fb (differential in t, integrated over Q^2 and y)
	def get_integrated_ssa(self, t_values, s, phase_space, **options):

		###### setting up parameters 
		method = options.get('method', 'gauss-legendre')
		x0 = options.get('x0', 0.1)
		points = options.get('points', 10)
		harmonic = options.get('harmonic', 'sin(phi)')

		# error checking
		assert method in ['gauss-legendre', 'riemann', 'mc'], f'Error: method = {method} not recognized'
		assert harmonic in ['sin(phi)', 'sin(2phi)'], f'Error: harmonic {harmonic} not recognized'
		#############################

		kinematics = Kinematics(s=s)
		integrated = {var: isinstance(space, list) for var, space in phase_space.items()}

		results = {i: [] for i in ['num', 'den', 'ssa']} 
		for t in t_values:

			kinematics.t = t

			# integrate using one of 3 methods: gaussian quadrature, riemann sums, or monte-carlo
			if method == 'gauss-legendre':

				# Gauss–Legendre nodes, weights on [-1,1]
				nodes, weights = leggauss(points)

				if integrated['y']:
					y_values, y_w = map_to_interval(nodes, weights, *phase_space['y'])
				else:
					y_values, y_w = [phase_space['y']], [1]

				result = {i: 0.0 for i in ['num', 'den']}
				for i, y in enumerate(y_values):
					kinematics.y = y

					if integrated['Q2']:
						Q2_min = phase_space['Q2'][0]
						Q2_max = x0 * s * y
						# print('hmmm')
						if Q2_max < Q2_min: continue
						# print('hmmm 2')
						Q2_values, Q2_w = map_to_interval(nodes, weights, Q2_min, Q2_max)
					else:
						Q2_values, Q2_w = [phase_space['Q2']], [1]

					for j, Q2 in enumerate(Q2_values):
						kinematics.Q = np.sqrt(Q2)
						x = Q2 / (s * y)
						kinematics.x = x
#
						# print(kinematics)

						weight_factor = y_w[i] * Q2_w[j]
						integral = self.get_ssa(kinematics, harmonic=harmonic)

						# print(integral)

						result['num'] += weight_factor * integral['num']
						result['den'] += weight_factor * integral['den']

			# elif method == 'riemann':

			# 	if integrated['z']: 
			# 		z_values = np.linspace(*phase_space['z'], points)
			# 		dz = z_values[1]-z_values[0]
			# 	else:
			# 		z_values, dz = [phase_space['z']], 1

			# 	if integrated['y']:
			# 		y_values = np.linspace(*phase_space['y'], points)
			# 		dy = y_values[1]-y_values[0]
			# 	else:
			# 		y_values, dy = [phase_space['y']], 1

			# 	if integrated['t']:
			# 		t_values = np.linspace(*phase_space['t'], points)
			# 		dt = t_values[1]-t_values[0]
			# 	else:
			# 		t_values, dt = [phase_space['t']], 1

			# 	result = 0.0

			# 	for i, y in enumerate(y_values):
			# 		kinematics.y = y

			# 		if integrated['Q2']:
			# 			Q2_max = min(x0 * s * y, phase_space['Q2'][1])
			# 			if Q2_max < phase_space['Q2'][0]: continue
			# 			Q2_values = np.linspace(phase_space['Q2'][0], Q2_max, points)
			# 			dQ2 = Q2_values[1]-Q2_values[0]
			# 		else:
			# 			Q2_values, dQ2 = [phase_space['Q2']], 1

			# 		for j, Q2 in enumerate(Q2_values):
			# 			kinematics.Q = np.sqrt(Q2)
			# 			x = Q2 / (s * y)
			# 			kinematics.x = x

			# 			if x > 0.01: continue

			# 			for k, z in enumerate(z_values):
			# 				if np.sqrt(Q2) * np.sqrt(z * (1 - z)) < r0: continue
			# 				kinematics.z = z

			# 				for l, t in enumerate(t_values):
			# 					kinematics.delta = np.sqrt(t)

			# 					weight_factor = dy * dQ2 * dz * dt
			# 					result += weight_factor * xsec_func(kinematics, weight=weight, diff='dy', kind=kind)

			# elif method == 'mc':
			
			# 	rng = np.random.default_rng()
			# 	ran_sum = 0
			# 	for i in range(points):
			# 		if integrated['y']:
			# 			kinematics.y = rng.uniform(*phase_space['y'])
			# 		else:
			# 			kinematics.y = phase_space['y']

			# 		if integrated['Q2']:
			# 			kinematics.Q = np.sqrt(rng.uniform(*phase_space['Q2']))
			# 		else:
			# 			kinematics.Q = np.sqrt(phase_space['Q2'])

			# 		kinematics.x = (kinematics.Q**2)/(s*kinematics.y)

			# 		if kinematics.x > 0.01: continue
			# 		if np.sqrt((kinematics.Q**2)*kinematics.z*(1-kinematics.z)) < r0: continue

			# 		ran_sum += xsec_func(kinematics, weight=weight, diff='dy', kind=kind)

			# 	box_volume = 1
			# 	for var in ['y', 't', 'Q2', 'z']:
			# 		if integrated[var]: box_volume *= phase_space[var][1]-phase_space[var][0]

			# 	result = ran_sum*(1/points)*box_volume


			results['num'].append(result['num']) 
			results['den'].append(result['den']) 
			results['ssa'].append(result['num']/result['den'])
		return results


	def ppdf_minus(self, x, Q2, x0=1.0):
    
		eta0 = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x0)
		eta0index = int(np.ceil(eta0/self.deta))+1
		eta = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x)
		
		jetai = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(1./x)/self.deta))
		jetaf = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(Q2/(x*self.lambdaIR))/self.deta))

		ppdf_minus=[0,0,0]

		for iflav, flav in enumerate(['u', 'd', 's']):
			for j in range(eta0index,jetai-1 +1):
				for i in range(0,j-eta0index-1 +1):
					ppdf_minus[iflav] += (self.deta**2) * self.dipoles[f'QNS{flav}'][i,j]

			for j in range(jetai-1+1,jetaf-1 +1):
				for i in range(j-jetai,j-eta0index-1 +1):
					ppdf_minus[iflav] += (self.deta**2) * self.dipoles[f'QNS{flav}'][i,j]

		ppdf_minus = np.array(ppdf_minus)*(1./(np.pi**2))
				
		return ppdf_minus
	

	def get_bands(self, reps, confid=68):
		bands = {}			
		bands['lower'] = np.percentile(reps, 0.5*(100-confid), axis=0)
		bands['upper'] = np.percentile(reps, 100 - 0.5*(100-confid), axis=0)
		bands['mean'] = np.mean(reps, axis=0)
		return bands




if __name__ == '__main__':

	ssa = PionSSA(fit_type='pp')
	# ssa.load_params('dis')
	# ssa.set_params(4)
	# print(ssa.ppdf_minus(0.01, 10))

	kins = Kinematics(s=(150**2), Q=np.sqrt(10), x=0.01, t=0.1)
	kins.y = (kins.Q**2)/(kins.x * kins.s)

	for t in [0.01, 0.02, 0.05, 0.1, 0.2]:
		kins.t = t
		sinphi = ssa.get_ssa(kins, harmonic='sin(phi)')
		sin2phi = ssa.get_ssa(kins, harmonic='sin(2phi)')
		print(sinphi['num'])
		print(sin2phi['num'])
		# sinphi = ssa.get_numerator(kins, harmonic='sin(phi)')
		# denom = ssa.get_denominator(kins)
		# print(sinphi, denom, sinphi/denom)


