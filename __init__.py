"""
=================
caustic_stack.py
=================

Stack individual clusters' phase spaces into an ensemble and run caustic technique

----------------------------------
Nick Kern, University of Michigan
nkern@umich.edu
Version 0.1
Updated: June, 2014
"""

# Load Modules
import numpy as np
import astropy.io.fits as fits
from numpy.linalg import norm
import matplotlib.pyplot as mp
import astrostats as astats
import sys, os
import time
import cPickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import numpy.random as npr
from astropy.stats import sigma_clip
from scipy import weave
from scipy.weave import converters
import cosmolopy.distance as cd
import DictEZ as ez
from causticpy import Caustic,CausticSurface,MassCalc
from calc_kcor import calc_kcor


class Data():
	"""
	Can be used as an open container for stacking data on itself
	As of now, data to-be-stacked must be as a list
	"""
	def check_varib(self,name):
		if name in self.__dict__:
			return True
		else:
			return False
	def append(self,DATA,keys=None):
		"""
		Takes DATA as a DICTIONARY:
		if the key already exists within self.__dict__ it APPENDS IT,
		if the key does not exist within self.__dict__ it ADDS IT.
		"""
		if keys == None:
			names = DATA.keys()
		else:
			names = keys
		# Iterate through variables defined in DATA
		for name in names:
			try:
				if self.check_varib(name) == True:
					self.__dict__[name].append(DATA[name])
				else:
					self.__dict__[name] = [DATA[name]]
			except KeyError:
				pass
	def extend(self,DATA,keys=None):
		"""
		Takes DATA as a DICTIONARY:
		if the key already exists within self.__dict__ it EXTENDS IT,
		if the key does not exist within self.__dict__ it ADDS IT.
		"""
                if keys == None:
                        names = DATA.keys()
                else:
                        names = keys
		# Iterate through variables defined in DATA
		for name in names:
			try:
				if self.check_varib(name) == True:
					try:
						self.__dict__[name].extend(list(DATA[name]))
					except:
						self.__dict__[name].extend([DATA[name]])
				else:
					try:
						self.__dict__[name] = list(DATA[name])
					except:
						self.__dict__[name] = [DATA[name]]
			except:
				pass
	def add(self,DATA,keys=None):
		"""
		Takes DATA as a DICTIONARY:
		if the key already exists within self.__dict__ it REPLACES IT,
		if the key does not exist within self.__dict__ it ADDS IT.
		"""
                if keys == None:
                        names = DATA.keys()
                else:
                        names = keys
		# Iterate through variables defined in DATA
		for name in names:
			try:
				try:
					len(DATA[name])
					self.__dict__[name] = np.array(DATA[name])
				except:
					self.__dict__[name] = np.array([DATA[name]])
			except:
				pass
	def clear(self):	
		"""
		Clears all variables in class
		"""
		self.__dict__.clear()
	def to_array(self,names,ravel=False):
		"""
		Turns data into arrays
		"""
		for i in names:
			try:
				if ravel == True:
					self.__dict__[i] = np.array(self.__dict__[i]).ravel()
				else:
					self.__dict__[i] = np.array(self.__dict__[i])
					if self.__dict__[i].shape[-1] == 1:
						self.__dict__[i] = self.__dict__[i].reshape(self.__dict__.shape[0:-1])
			except:
				pass
	def upper(self,names=None):
		"""
		Turns all keys of Data into upper case, or those that are fed via names
		"""
		if names == None:
			names = self.__dict__.keys()
		for name in names:
			try: self.__dict__[name.upper()] = self.__dict__.pop(name)
			except: pass
	def lower(self,names=None):
		"""
		Turns all keys of Data into lower case, or those that are fed via naems
		"""
		if names == None:
			names = self.__dict__.keys()
		for name in names:
			try: self.__dict__[name.lower()] = self.__dict__.pop(name)
			except: pass

class Stack(object):
	"""
	The main class that runs the caustic technique over a stacking routine
	"""
	def __init__(self,varib):
		# Update Dictionary with Variables and Caustic Technique
		self.__dict__.update(varib)
		self.C = Caustic()
		self.U = Universal(varib)

	def run_caustic(self,rvalues,vvalues,R200,HVD,clus_z=0,shiftgap=False,mirror=True,ensemble=True):
		"""
		Calls causticpy's run_caustic function
		"""
		# Feed caustic dummy vertical array
		length = len(rvalues)
		dummy = np.zeros(length).reshape(length,1)

		# Run Caustic
		self.C.run_caustic(dummy,gal_r=rvalues,gal_v=vvalues,r200=R200,clus_z=clus_z,gapper=shiftgap,mirror=mirror,edge_perc=self.edge_perc,q=self.q,rlimit=self.r_limit*R200,vlimit=self.v_limit,H0=self.H0)
		#self.C.run_caustic(dummy,gal_r=rvalues,gal_v=vvalues,r200=R200,clus_z=clus_z,clus_vdisp=HVD,gapper=shiftgap,mirror=mirror,edge_perc=self.edge_perc,q=self.q,rlimit=self.r_limit*R200,vlimit=self.v_limit,H0=self.H0)


	def caustic_stack(self,Rdata,Vdata,HaloID,HaloData,stack_num,
				ens_shiftgap=True,gal_reduce=True,stack_raw=False,est_v_center=False,
				feed_mags=True,G_Mags=None,R_Mags=None,I_Mags=None):
		"""
		-- Takes an array of individual phase spaces and stacks them, then runs 
		   a caustic technique over the ensemble and/or individual phase spaces.
		-- Returns a dictionary

		-- Relies on a few parameters to be defined outside of the function:
			self.gal_num - Equivalent to Ngal: number of galaxies to take per cluster to then be stacked
			self.line_num - Equivalent to Nclus: number of clusters to stack into one ensemble cluster
			self.scale_data - Scale r data by R200 while stacking, then un-scale by BinR200
			self.run_los - run caustic technique over individual cluster (aka line-of-sight)
			self.avg_meth - method by which averaging of Bin Properties should be, 'mean' or 'median' or 'biweight' etc..
			self.mirror - mirror phase space before solving for caustic?
			Including others... see RunningTheCode.pdf for a list
			* These parameters should be fed to initialization of Stack() class as a dictionary, for ex:
				variables = {'run_los':False, ...... }
				S = Stack(variables)

		"rdata" - should be a 2 dimensional array with individual phase spaces as rows
		"vdata" - should be a 2 dimensional array with individual phase spaces as rows
				ex. rdata[0] => 0th phase space data
		'HaloID' : 1 dimensional array containing Halo Identification Numbers, len(HaloID) == len(rdata)
		'HaloData' : 2 dimensional array containing M200, R200, HVD of Halos, with unique halos as columns
		'stack_num' : number of individual clusters to stack into the one ensemble

		'ens_shiftgap' - do a shiftgapper over final ensemble phase space?
		'gal_reduce' - before caustic run, reduce ensemble phase space to Ngal gals within R200?
		'stack_raw' - don't worry about limiting or building the phase space, just stack the Rdata and Vdata together as is
		'est_v_center' - take median of Vdata for new velocity center

		'feed_gal_mags' - feed magnitudes for individual galaxies, must feed all three mags if True

		-- 'ens' stands for ensemble cluster
		-- 'ind' stands for individual cluster

		-- Uppercase arrays contain data for multiple clusters,
			lowercase arrays contain data for 1 cluster
		"""

		# Define a container for holding stacked data, D
		D = Data()

		# Assign some parameters to Class scope
		self.__dict__.update(ez.create(['stack_raw','feed_mags','gal_reduce','ens_shiftgap'],locals()))

		# New Velocity Center
		if est_v_center == True:
			v_offset = astats.biweight_location(vdata[np.where(rdata < 1.0)])
			vdata -= v_offset

		# Unpack HaloData
		if HaloData == None:
			self.fed_halo_data = False
			# Estimate R200
			R200 = []
			HVD = []
			for i in range(stack_num):
				R200.append(np.exp(-1.86)*len(np.where((R_Mags[i] < -19.55) & (Rdata[i] < 1.0) & (np.abs(Vdata[i]) < 3500))[0])**0.51)
				HVD.append(astats.biweight_location(Vdata[i][np.where((Rdata[i] < 1.0)&(np.abs(Vdata[i])<4000))]))
			R200 = np.array(R200)
			HVD = np.array(HVD)

			if self.avg_meth == 'mean': BinR200 = np.mean(R200); BinHVD = np.mean(HVD)
			elif self.avg_meth == 'median': BinR200 = np.median(R200); BinHVD = np.median(HVD)
			D.add({'BinR200':BinR200,'BinHVD':BinHVD})			

		else:
			self.fed_halo_data = True
			M200,R200,HVD = HaloData
			if self.avg_meth == 'mean':
				BinM200 = np.mean(M200)
				BinR200 = np.mean(R200)
				BinHVD = np.mean(HVD)
			elif self.avg_meth == 'median':
				BinM200 = np.median(M200)
				BinR200 = np.median(R200)
				BinHVD = np.median(HVD)
			# Append to Data
			D.add({'BinM200':BinM200,'BinR200':BinR200,'BinHVD':BinHVD})	

		# Create Dummy Variables for Magnitudes if necessary
		if self.feed_mags == False:
			G_Mags,R_Mags,I_Mags = [],[],[]
			for i in range(stack_num):
				G_Mags.append([None]*len(Rdata[i]))
				R_Mags.append([None]*len(Rdata[i]))
				I_Mags.append([None]*len(Rdata[i]))
			G_Mags,R_Mags,I_Mags = np.array(G_Mags),np.array(R_Mags),np.array(I_Mags)

		# Create galaxy identification arrays
		ENS_gal_id,ENS_clus_id,IND_gal_id = [],[],[]
		gal_count = 0
		for i in range(stack_num):
			ENS_gal_id.append(np.arange(gal_count,gal_count+len(Rdata[i])))
			ENS_clus_id.append(np.array([HaloID[i]]*len(Rdata[i]),int))
			IND_gal_id.append(np.arange(len(Rdata[i])))
		ENS_gal_id,ENS_clus_id,ENS_gal_id = np.array(ENS_gal_id),np.array(ENS_clus_id),np.array(IND_gal_id)

		# Iterate through phase spaces
		for self.l in range(stack_num):

			# Limit Phase Space
			if self.stack_raw == False:
				r,v,ens_gal_id,ens_clus_id,ind_gal_id,gmags,rmags,imags,samp_size = self.U.limit_gals(Rdata[self.l],Vdata[self.l],ENS_gal_id[self.l],ENS_clus_id[self.l],IND_gal_id[self.l],G_Mags[self.l],R_Mags[self.l],I_Mags[self.l],R200[self.l])

			# Build Ensemble and LOS Phase Spaces
			if self.stack_raw == False:
				ens_r,ens_v,ens_gal_id,ens_clus_id,ens_gmags,ens_rmags,ens_imags,ind_r,ind_v,ind_gal_id,ind_gmags,ind_rmags,ind_imags = self.U.build(r,v,ens_gal_id,ens_clus_id,ind_gal_id,gmags,rmags,imags,R200[self.l])

			# If Scale data before stack is desired
			if self.scale_data == True:
				ens_r /= R200[self.l]

			# Stack Ensemble Data by extending to Data() container
			names = ['ens_r','ens_v','ens_gmags','ens_rmags','ens_imags','ens_gal_id','ens_clus_id']
			D.extend(ez.create(names,locals()))

			if self.run_los == True:
		
				# Create Data Block
				ind_data = np.vstack([ind_r,ind_v,ind_gal_id,ind_gmags,ind_rmags,ind_imags])

				# Sort by Rmag 
				bright = np.argsort(ind_rmags)
				ind_data = ind_data.T[bright].T
				ind_r,ind_v,ind_gal_id,ind_gmags,ind_rmags,ind_imags = ind_data
	
				# Reduce phase space
				if self.stack_raw == False and self.gal_reduce == True:
					within = np.where(ind_r <= R200[self.l])[0]
					end = within[:self.gal_num + 1][-1]
					ind_data = ind_data.T[:end].T
					ind_r,ind_v,ind_gal_id,ind_gmags,ind_rmags,ind_imags = ind_data
	
				# Calculate individual HVD
				# Pick out gals within r200
				within = np.where(ind_r <= R200[self.l])[0]
				gal_count = len(within)
				if gal_count <= 3:
					'''biweightScale can't take less than 4 elements'''
					# Calculate hvd with numpy std of galaxies within r200 (b/c this is quoted richness)
					ind_hvd = np.std(np.copy(ind_v)[within])
				else:
					# Calculate hvd with astStats biweightScale (see Beers 1990)
					try:
						ind_hvd = astats.biweight_midvariance(np.copy(ind_v)[within])
					# Sometimes divide by zero error in biweight function for low gal_num
					except ZeroDivisionError:
						print 'ZeroDivisionError in biweightfunction'
						print 'ind_v[within]=',ind_v[within]
						ind_hvd = np.std(np.copy(ind_v)[within])

				# If run_los == True, run Caustic Technique on individual cluster
				self.U.print_separation('# Running Caustic for LOS '+str(self.l),type=2)
				self.run_caustic(ind_r,ind_v,R200[self.l],ind_hvd,mirror=self.mirror)
				ind_caumass = np.array([self.C.M200_fbeta])
				ind_caumass_est = np.array([self.C.Mass2.M200_est])
				ind_edgemass = np.array([self.C.M200_edge])
				ind_edgemass_est = np.array([self.C.MassE.M200_est])
				ind_causurf = np.array(self.C.caustic_profile)
				ind_nfwsurf = np.array(self.C.caustic_fit)
				ind_edgesurf = np.array(self.C.caustic_edge)

			# Append Individual Cluster Data
			names = ['ind_r','ind_v','ind_gal_id','ind_gmags','ind_rmags','ind_imags',
				'ind_hvd','ind_caumass','ind_caumass_est','ind_edgemass','ind_edgemass_est',
				'ind_causurf','ind_nfwsurf']
			D.append(ez.create(names,locals()))


		# Turn Ensemble into Arrays
		names = ['ens_r','ens_v','ens_gal_id','ens_clus_id','ens_gmags','ens_rmags','ens_imags','ens_caumass','ens_caumass_est','ens_edgemass','ens_edgemass_est','ens_causurf','ens_nfwsurf','ens_edgesurf']
		D.to_array(names,ravel=True)

		# Re-scale data if scale_data == True:
		if self.scale_data == True:
			D.ens_r *= BinR200

		# Create Ensemble Data Block
		D.ens_data = np.vstack([D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags])

		# Shiftgapper for Interloper Treatment
		if self.stack_raw == False and self.ens_shiftgap == True:
			self.D = D
			D.ens_data = self.C.shiftgapper(D.ens_data.T).T
			D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags = D.ens_data

		# Sort by R_Mag
		bright = np.argsort(D.ens_rmags)
		D.ens_data = D.ens_data.T[bright].T
		D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags = D.ens_data

		# Reduce System Down to gal_num richness within BinR200
		if self.stack_raw and self.gal_reduce == True:
			within = np.where(D.ens_r <= BinR200)[0]
			end = within[:self.gal_num*self.line_num + 1][-1]
			D.ens_data = D.ens_data.T[:end].T
			D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags = D.ens_data

		# Calculate Ensemble Velocity Dispersion for galaxies within R200
		ens_hvd = astats.biweight_midvariance(np.copy(D.ens_v)[np.where(D.ens_r<=BinR200)])

		# Run Caustic Technique!
		try: self.U.print_separation('# Running Caustic on Ensemble '+str(self.j),type=2)
		except: pass
		self.run_caustic(D.ens_r,D.ens_v,BinR200,BinHVD,mirror=self.mirror)
		ens_caumass = np.array([self.C.M200_fbeta])
		ens_caumass_est = np.array([self.C.Mass2.M200_est])
		ens_edgemass = np.array([self.C.M200_edge])
		ens_edgemass_est = np.array([self.C.MassE.M200_est])
		ens_causurf = np.array(self.C.caustic_profile)
		ens_nfwsurf = np.array(self.C.caustic_fit)
		ens_edgesurf = np.array(self.C.caustic_edge)

		# Other Arrays
		x_range = self.C.x_range

		# Append Data
		names = ['ens_caumass','ens_hvd','ens_caumass_est','ens_edgemass','ens_edgemass_est','ens_causurf','ens_nfwsurf','ens_edgesurf','x_range']
		D.add(ez.create(names,locals()))

		# Turn Individual Data into Arrays
		if self.run_los == True:
			names = ['ind_caumass','ind_caumass_est','ind_edgemass','ind_edgemass_est','ind_hvd']
			D.to_array(names,ravel=True)
			names = ['ind_r','ind_v','ind_gal_id','ind_gmags','ind_rmags','ind_imags','ind_causurf','ind_nfwsurf','ind_edgesurf']
			D.to_array(names)	

		# Output Data
		return D.__dict__


class Universal(object):
	"""
	Other functions that can be used in the building and set-up of data
	"""
	
	def __init__(self,varib):
		self.__dict__.update(varib)
		self.C = Caustic()


	def build(self,r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,r200):
		"""
		This function builds the ensemble and individual cluster phase spaces, depending on fed parameters
		method 0 : top Ngal brightest
		method 1 : random Ngal within top Ngal*~10 brightest
		Interloper treatment done with ShiftGapper
		r : radius
		v : velocity
		en_gal_id : unique id for each ensemble galaxy
		en_clus_id : id for each galaxy relating back to its original parent halo
		ln_gal_id : unique id for each individual cluster galaxy
		rmags : SDSS r magnitude
		imags : SDSS i magnitude
		gmags : SDSS g magnitude
		halodata : 2 dimensional array, with info on halo properties
		- r200
		"""
		# Sort galaxies by r Magnitude
		bright = np.argsort(rmags)
		r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags = r[bright],v[bright],en_gal_id[bright],en_clus_id[bright],ln_gal_id[bright],gmags[bright],rmags[bright],imags[bright]

		if self.method_num == 0:
			en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags,ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags = self.build_method_0(r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,r200)

		elif self.method_num == 1:
			en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags,ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags,samp_size = self.build_method_1(r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,r200)

		else:
			print 'No Build...?'

		return en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags,ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags


	def build_method_0(self,r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,r200):
		'''Picking top brightest galaxies, such that there are gal_num galaxies within r200'''

		gal_num = self.gal_num

		## ------------------------------- ##
		## Build Ensemble (en => ensemble) ##
		## ------------------------------- ##
	
		# define indicies of galaxies within r200
		within = np.where(r<r200)[0]
		# pick out gal_num 'th index in list, (include extra to counter shiftgapper's possible dimishement of richness)
		if gal_num < 10:
			excess = gal_num * 3.0 / 5.0				# galaxy excess to counter shiftgapper
		else:
			excess = gal_num / 5.0
		end = within[:gal_num + excess + 1][-1]		# instead of indexing I am slicing b/c of chance of not enough gals existing...	
		# Choose Ngals within r200
		if self.init_clean == True:
			excess *= 2.0				# make excess a bit larger than previously defined
			end = within[:gal_num + excess + 1][-1]
			r2,v2,en_gal_id,en_clus_id,gmags2,rmags2,imags2 = self.C.shiftgapper(np.vstack([r[:end],v[:end],en_gal_id[:end],en_clus_id[:end],gmags[:end],rmags[:end],imags[:end]]).T).T # Shiftgapper inputs and outputs data as transpose...
			within = np.where(r2<r200)[0]	# re-calculate within array with new sample
			excess = gal_num / 5.0
			end = within[:gal_num + excess + 1][-1]
			# Append to ensemble array
			en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags = r2[:end],v2[:end],en_gal_id[:end],en_clus_id[:end],gmags2[:end],rmags2[:end],imags2[:end]
		else:
			en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags = r[0:end],v[0:end],en_gal_id[:end],en_clus_id[:end],gmags[0:end],rmags[0:end],imags[0:end]


		## ----------------------------------------- ##
		## Build Line of Sight (ln => line of sight) ##
		## ----------------------------------------- ##

		# define indicies of galaxies within r200
		within = np.where(r<r200)[0]
		# pick out gal_num 'th index in list, (include extra to counter shiftgapper's possible dimishement of richness)
		if gal_num < 10:
			excess = gal_num * 3.0 / 5.0				# galaxy excess to counter shiftgapper
		else:
			excess = gal_num / 5.0
		end = within[:gal_num + excess + 1][-1]		# instead of indexing I am slicing b/c
		# shiftgapper on line of sight
		r2,v2,ln_gal_id,gmags2,rmags2,imags2 = self.C.shiftgapper(np.vstack([r[:end],v[:end],ln_gal_id[:end],gmags[:end],rmags[:end],imags[:end]]).T).T

		# Sort by rmags
		sort = np.argsort(rmags2)
		r2,v2,ln_gal_id,gmags2,rmags2,imags2 = r2[sort],v2[sort],ln_gal_id[sort],gmags2[sort],rmags2[sort],imags2[sort]

		ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags = r2,v2,ln_gal_id,gmags2,rmags2,imags2

		# Done! Now we have en_r and ln_r arrays, which will either be stacked (former) or put straight into Caustic technique (latter)
		return en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags,ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags


	def build_method_1(self,r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,r_crit200):
		'''Randomly choosing bright galaxies until gal_num galaxies are within r200'''

		gal_num = self.gal_num
		
		# reduce size of sample to something reasonable within magnitude limits
		sample = gal_num * 25				# arbitrary coefficient, see sites page post Apr 24th, 2013 for more info
		r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags = r[:sample],v[:sample],en_gal_id[:sample],en_clus_id[:sample],ln_gal_id[:sample],gmags[:sample],rmags[:sample],imags[:sample]
		samp_size = len(r)				# actual size of sample (might be less than gal_num*25)
		self.samp_size = samp_size

		# create random numbered array for galaxy selection
		if gal_num < 10:				# when gal_num < 10, has trouble hitting gal_num richness inside r200
			excess = gal_num * 4.0 / 5.0
		else:
			excess = gal_num * 2.0 / 5.0

		samp_num = gal_num + excess			# sample size of randomly generated numbers, start too low on purpose, then raise in loop
		loop = True					# break condition
		while loop == True:				# see method 0 comments on variables such as 'excess' and 'within' and 'end'
			for it in range(3):			# before upping sample size, try to get a good sample a few times
				rando = npr.randint(0,samp_size,samp_num)
				within = np.where(r[rando]<=r_crit200)[0]
				if len(within) >= gal_num + excess:
					loop = False
			if len(within) < gal_num + excess:
				samp_num += 2

		### Build Ensemble ###
		if self.init_clean == True:

			r2,v2,en_gal_id,en_clus_id,gmags2,rmags2,imags2 = self.C.shiftgapper(np.vstack([r[rando],v[rando],en_gal_id[rando],en_clus_id[rando],gmags[rando],rmags[rand],imags[rando]]).T).T
			within = np.where(r2<r_crit200)[0]
			excess = gal_num / 5.0
			end = within[:gal_num + excess + 1][-1]
			# Append to ensemble array
			en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags = r2[:end],v2[:end],en_gal_id[:end],en_clus_id[:end],gmags2[:end],rmags2[:end],imags2[:end]
		else:
			excess = gal_num / 5.0
			end = within[:gal_num + excess + 1][-1]
			en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags = r[rando][:end],v[rando][:end],en_gal_id[rando][:end],en_clus_id[rando][:end],gmags[rando][:end],rmags[rando][:end],imags[rando][:end]

		### Build LOS ###
		# Set Sample
		if gal_num < 10:
			excess = gal_num * 4.0 / 5.0
		else:
			excess = gal_num * 2.0 / 5.0
		try:
			end = within[:gal_num + excess + 1][-1]
			# shiftgapper
			r2,v2,ln_gal_id2,gmags2,rmags2,imags2 = self.C.shiftgapper(np.vstack([r[rando][:end],v[rando][:end],ln_gal_id[rando][:end],gmags[rando][:end],rmags[rando][:end],imags[rando][:end]]).T).T
			# sort by rmags
			sort = np.argsort(rmags2)
			r2,v2,ln_gal_id2,gmags2,rmags2,imags2 = r2[sort],v2[sort],ln_gal_id2[sort],gmags2[sort],rmags2[sort],imags2[sort]
			# feed back gal_num gals within r200
			within = np.where(r2<r_crit200)[0]
			end = within[:gal_num + 1][-1]
			richness = len(within)
		except IndexError:
			print '****RAISED INDEX ERROR on LOS Building****'
			richness = 0		

		# Make sure gal_num richness inside r200
		run_time = time.asctime()
		j = 0
		while richness < gal_num:
			## Ensure this loop doesn't get trapped forever
			duration = (float(time.asctime()[11:13])*3600+float(time.asctime()[14:16])*60+float(time.asctime()[17:19]))-(float(run_time[11:13])*3600+float(run_time[14:16])*60+float(run_time[17:19]))
			if duration > 30:
				print "****Duration exceeded 30 seconds in LOS Buiding, manually broke loop****"
				break
			##
			j += 1
			loop = True
			while loop == True:				
				for j in range(3):			
					rando = npr.randint(0,samp_size,samp_num)
					within = np.where(r[rando]<=r_crit200)[0]
					if len(within) >= gal_num + excess:
						loop = False
				if len(within) < gal_num + excess:
					samp_num += 2
			try:
				end = within[:gal_num + excess + 1][-1]
				r2,v2,ln_gal_id2,gmags2,rmags2,imags2 = self.C.shiftgapper(np.vstack([r[rando][:end],v[rando][:end],ln_gal_id[rando][:end],gmags[rando][:end],rmags[rando][:end],imags[rando][:end]]).T).T
				within = np.where(r2<r_crit200)[0]
				end = within[:gal_num + 1][-1]
				richness = len(within)
			except IndexError:
				print '**** Raised Index Error on LOS Building****'
				richness = 0

			if j >= 100:
				print 'j went over 100'
				break

		ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags = r2[:end],v2[:end],ln_gal_id2[:end],gmags2[:end],rmags2[:end],imags2[:end]
		# Done! Now we have en_r and ln_r arrays (ensemble and line of sight arrays)
		
		return en_r,en_v,en_gal_id,en_clus_id,en_gmags,en_rmags,en_imags,ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags,samp_size


	def rand_pos(self,distance):
	        '''Picks a random position for the observer a given distance away from the center'''
		theta = npr.normal(np.pi/2,np.pi/4)
		phi = npr.uniform(0,2*np.pi)
		x = np.sin(theta)*np.cos(phi)
		y = np.sin(theta)*np.sin(phi)
		z = np.cos(theta)
	
		unit = np.array([x,y,z])/(x**2+y**2+z**2)**(.5)
		# move the position a random 'distance' Mpc away
	        return distance*unit


	def limit_gals(self,r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,r200):
		''' Sort data by magnitude, and elimite values outside phase space limits '''
		# Sort by ascending r magnitude (bright to dim)
		sorts = np.argsort(rmags)
		r = r[sorts]
		v = v[sorts]
		en_gal_id = en_gal_id[sorts]
		en_clus_id = en_clus_id[sorts]
		ln_gal_id = ln_gal_id[sorts]
		gmags = gmags[sorts]
		rmags = rmags[sorts]
		imags = imags[sorts]

		# Limit Phase Space
		sample = np.where( (r < r200*self.r_limit) & (v > -self.v_limit) & (v < self.v_limit) )[0] 
		r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags = r[sample],v[sample],en_gal_id[sample],en_clus_id[sample],ln_gal_id[sample],gmags[sample],rmags[sample],imags[sample]
		samp_size = len(sample)

		# Eliminate galaxies w/ mag = 99.
		cut = np.where((gmags!=99)&(rmags!=99)&(imags!=99))[0]
		r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags = r[cut],v[cut],en_gal_id[cut],en_clus_id[cut],ln_gal_id[cut],gmags[cut],rmags[cut],imags[cut]
		samp_size = len(cut)

		return r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,samp_size


	def Bin_Calc(self,M200,R200,HVD):
		'''
		This function does pre-technique binning analysis
		'''
		# Choose Averaging Method
		if self.avg_meth == 'median':
			avg_method = np.median
		elif self.avg_meth == 'mean':
			avg_method = np.mean
		else:
			print 'Average Method for Bin is Mean()'
			avg_method = np.mean

		# Calculate Bin R200 and Bin HVD, use median
		BIN_M200,BIN_R200,BIN_HVD = [],[],[]
		for i in range(len(M200)/self.line_num):
			BIN_M200.append( avg_method( M200[i*self.line_num:(i+1)*self.line_num] ) )
			BIN_R200.append( avg_method( R200[i*self.line_num:(i+1)*self.line_num] ) )
			BIN_HVD.append( avg_method( HVD[i*self.line_num:(i+1)*self.line_num] ) )

		BIN_M200,BIN_R200,BIN_HVD = np.array(BIN_M200),np.array(BIN_R200),np.array(BIN_HVD)

		return BIN_M200,BIN_R200,BIN_HVD


	def get_3d(self,Gal_P,Gal_V,ens_gal_id,los_gal_id,stack_range,ens_num,self_stack,j):
		'''
		This function recovers the 3D positions and velocities of the galaxies in the ensemble and los phase space.
		'''

		if self_stack == True:
			# Create fully concatenated arrays to draw ensemble data from
			GPX3D,GPY3D,GPZ3D = Gal_P[j][0],Gal_P[j][1],Gal_P[j][0]
			GVX3D,GVY3D,GVZ3D = Gal_V[j][0],Gal_V[j][1],Gal_V[j][0]
		
			# Recover ensemble 3D data
			ens_gpx3d,ens_gpy3d,ens_gpz3d = GPX3D[ens_gal_id],GPY3D[ens_gal_id],GPZ3D[ens_gal_id]
			ens_gvx3d,ens_gvy3d,ens_gvz3d = GVX3D[ens_gal_id],GVY3D[ens_gal_id],GVZ3D[ens_gal_id]

			# Recover line_of_sight 3D data	
			los_gpx3d,los_gpy3d,los_gpz3d = np.array(map(lambda x: GPX3D[x],los_gal_id)),np.array(map(lambda x: GPY3D[x],los_gal_id)),np.array(map(lambda x: GPZ3D[x],los_gal_id))
			los_gvx3d,los_gvy3d,los_gvz3d = np.array(map(lambda x: GVX3D[x],los_gal_id)),np.array(map(lambda x: GVY3D[x],los_gal_id)),np.array(map(lambda x: GVZ3D[x],los_gal_id))

		else:	
			# Create fully concatenated arrays to draw ensemble data from
			[BIN_GPX3D,BIN_GPY3D,BIN_GPZ3D] = map(np.concatenate,Gal_P[j*self.line_num:(j+1)*self.line_num].T)
			[BIN_GVX3D,BIN_GVY3D,BIN_GVZ3D] = map(np.concatenate,Gal_V[j*self.line_num:(j+1)*self.line_num].T)

			# Recover ensemble 3D data	
			ens_gpx3d,ens_gpy3d,ens_gpz3d = BIN_GPX3D[ens_gal_id],BIN_GPY3D[ens_gal_id],BIN_GPZ3D[ens_gal_id]
			ens_gvx3d,ens_gvy3d,ens_gvz3d = BIN_GVX3D[ens_gal_id],BIN_GVY3D[ens_gal_id],BIN_GVZ3D[ens_gal_id]

			# Recover line_of_sight 3D data
			los_gpx3d,los_gpy3d,los_gpz3d = [],[],[]
			los_gvx3d,los_gvy3d,los_gvz3d = [],[],[]
			for i,k in zip( np.arange(j*self.line_num,(j+1)*self.line_num), np.arange(self.line_num) ):
				los_gpx3d.append(Gal_P[i][0][los_gal_id[k]])
				los_gpy3d.append(Gal_P[i][1][los_gal_id[k]])
				los_gpz3d.append(Gal_P[i][2][los_gal_id[k]])
				los_gvx3d.append(Gal_V[i][0][los_gal_id[k]])
				los_gvy3d.append(Gal_V[i][1][los_gal_id[k]])
				los_gvz3d.append(Gal_P[i][2][los_gal_id[k]])

			los_gpx3d,los_gpy3d,los_gpz3d = np.array(los_gpx3d),np.array(los_gpy3d),np.array(los_gpz3d)
			los_gvx3d,los_gvy3d,los_gvz3d = np.array(los_gvx3d),np.array(los_gvy3d),np.array(los_gvz3d)

		return np.array([ens_gpx3d,ens_gpy3d,ens_gpz3d]),np.array([ens_gvx3d,ens_gvy3d,ens_gvz3d]),np.array([los_gpx3d,los_gpy3d,los_gpz3d]),np.array([los_gvx3d,los_gvy3d,los_gvz3d])


	def line_of_sight(self,gal_p,gal_v,halo_p,halo_v):
		'''Line of Sight Calculations to mock projected data, if given 3D data'''
		# Pick Position
		new_pos = self.rand_pos(30)
		new_pos += halo_p 

		# New Halo Information
		halo_dist = ((halo_p[0]-new_pos[0])**2 + (halo_p[1]-new_pos[1])**2 + (halo_p[2]-new_pos[2])**2)**0.5
		halo_pos_unit = np.array([halo_p[0]-new_pos[0],halo_p[1]-new_pos[1],halo_p[2]-new_pos[2]]) / halo_dist
		halo_vlos = np.dot(halo_pos_unit, halo_v)

		# New Galaxy Information
		gal_p = np.array(gal_p)
		gal_v = np.array(gal_v)
		gal_dist = ((gal_p[0]-new_pos[0])**2 + (gal_p[1]-new_pos[1])**2 + (gal_p[2]-new_pos[2])**2)**0.5
		gal_vlos = np.zeros(gal_dist.size)
		gal_pos_unit = np.zeros((3,gal_dist.size))	#vector from new_p to gal	
		n = gal_dist.size

		# Line of sight
		code = """
		int u,w;
		for (u=0;u<n;++u){
		for(w=0;w<3;++w){
		gal_pos_unit(w,u) = (gal_p(w,u)-new_pos(w))/gal_dist(u);
		}
		gal_vlos(u) = gal_pos_unit(0,u)*gal_v(0,u)+gal_pos_unit(1,u)*gal_v(1,u)+gal_pos_unit(2,u)*gal_v(2,u);
		}
		"""
		fast = weave.inline(code,['gal_pos_unit','n','gal_dist','gal_vlos','gal_v','new_pos','gal_p'],type_converters=converters.blitz,compiler='gcc')
		angles = np.arccos(np.dot(halo_pos_unit,gal_pos_unit))
		r = angles*halo_dist
		#v_pec = gal_vlos-halo_vlos*np.dot(halo_pos_unit,gal_pos_unit)
		z_clus_cos = self.H0*halo_dist/self.c
		z_clus_pec = halo_vlos/self.c
		z_clus_obs = (1+z_clus_pec)*(1+z_clus_cos)-1
		z_gal_cos = self.H0*gal_dist/self.c
		z_gal_pec = gal_vlos/self.c
		z_gal_obs = (1+z_gal_pec)*(1+z_gal_cos)-1
		v = self.c*(z_gal_obs-z_clus_obs)/(1+z_clus_obs)
		#gal_vdisp3d[i] = np.sqrt(astStats.biweightScale(gal_v[0][np.where(gal_radius<=HaloR200[i])]-Halo_V[0],9.0)**2+astStats.biweightScale(gal_v[1][np.where(gal_radius<=HaloR200[i])]-Halo_V[1],9.0)**2+astStats.biweightScale(gal_v[2][np.where(gal_radius<=HaloR200[i])]-Halo_V[2],9.0)**2)/np.sqrt(3)
		#print 'MY VELOCITY OF GALAXIES', gal_vdisp3d[i]
#		particle_vdisp3d[i] = HVD*np.sqrt(3)
#		gal_rmag_new = gal_abs_rmag# + 5*np.log10(gal_dist*1e6/10.0)

		return r, v, np.array(new_pos)


	def app2abs(self,m_app,color,z,photo_band,color_band):
		'''
		takes apparent magnitude of a galaxy in some photometric band (SDSS g or r or i)
		and converts it to an absolute magnitude via distance modulus and k correction
		M_abs = m_app - Dist_Mod - K_corr
		takes:
			m_app		: float, apparent magnitude of galaxy
			color		: float, color of galaxy between two SDSS bands
			z		: float, total redshift of galaxy
			photo_band	: str, SDSS band for magnitude. Ex: 'g' or 'r' or 'i'
			color_band	: str, color between two SDSS bands. Ex: 'g - r' or 'g - i' or 'r - i', etc..
		'''
		dm = self.distance_mod(z)
		K_corr = calc_kcor(photo_band,z,color_band,color)
		return m_app - dm - K_corr


	def distance_mod(self,z):
		ang_d,lum_d = self.C.zdistance(z,self.H0)
		dm = 5.0 * np.log10(lum_d*1e6 / 10.0 )
		return dm


	def print_varibs(self,names,dictionary):
		print '## Variables Defined in the Run'
		print '-'*50
		for i in names:
			try:
				if i=='':
					print ''
				print i+'\r\t\t\t= '+str(dictionary[i])
			except:
				pass
		print '-'*50

	def print_separation(self,text,type=1):
		if type==1:
			print ''
			print '-'*60
			print str(text)
			print '-'*60
		elif type==2:
			print ''
			print str(text)
			print '-'*30
		return












