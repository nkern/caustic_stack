"""
=================
caustic_stack.py
=================

Stack individual clusters' phase spaces and run caustic technique

"""
# Load Modules
import numpy as np
import astropy.io.fits as fits
from numpy.linalg import norm
import matplotlib.pyplot as mp
import astStats
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


class Data():
	"""
	Can be used as an open container for stacking data on itself
	"""
	def check_varib(self,name):
		if name in self.__dict__:
			return True
		else:
			return False
	def add(self,DATA):
		"""
		Takes DATA as a DICTIONARY and appends its keys to variables of the same name in the self.__dict__ 
		Each variable in DATA should be a 1 dimensional array
		All data types should be input and output in numpy arrays
		"""
		# Iterate through variables defined in DATA
		for name in DATA:
			if self.check_varib(name) == True:
				self.__dict__[name] = np.append(self.__dict__[name],DATA[name])
			else:
				self.__dict__[name] = np.copy(DATA[name]) 
	def clear(self):
		"""
		Clears all variables in class
		"""
		self.__dict__.clear()



class Stack(object):
	"""
	The main class that runs the caustic technique over a stacking routine
	"""
	def __init__(self,varib):
		# Update Dictionary with Variables and Caustic Technique
		self.__dict__.update(varib)
		self.C = Caustic()
		self.U = Universal(varib)

	def run_caustic(self,rvalues,vvalues,R200,HVD,clus_z=0,shiftgap=False,mirror=True):
		"""
		Calls causticpy's run_caustic function
		"""
		self.U.print_separation("## Working on Halo Number: "+str(self.l),type=2)
		
		# Feed caustic dummy vertical array
		length = len(rvalues)
		dummy = np.zeros(length).reshape(length,1)

		# Run Caustic
		self.C.run_caustic(dummy,gal_r=rvalues,gal_v=vvalues,r200=R200,clus_z=clus_z,clus_vdisp=HVD,gapper=shiftgap,mirror=mirror)



	def caustic_stack(self,ens_data,HaloID,HaloData,BinData,stack_num,ind_data=[None]*5,ens_shiftgap=True,ens_reduce=True,run_los=False):
		"""
		-- Takes an array of individual phase spaces and stacks them, then runs 
		   a caustic technique over the ensemble and/or individual phase spaces.
		-- Returns a dictionary
		-- ens_data and ind_data should be a 3 dimensional tuple or array

		'ens_r' : Should be an object with 'stack_num' rows, each with possibly variable lengths.
		'ens_v' : Should be an object with 'stack_num' rows, each with varying or identical lengths
		'HaloID' : 1 dimensional array containing Halo Identification Numbers
		'HaloData' : 2 dimensional array containing M200, R200, HVD of Halos
		'stack_num' : number of individual clusters to stack into the one ensemble
		
		-- 'en' or 'ens' stands for ensemble cluster
		-- 'in' or 'ind' stands for individual cluster
		"""
		# Unpack ens_data
		ens_r,ens_v,ens_gmags,ens_rmags,ens_imags = ens_data
	
		# Unpack ind_data
		ind_r,ind_v,ind_gmags,ind_rmags,ind_imags = ind_data

		# Unpack HaloData
		M200,R200,HVD = HaloData
		BinM200,BinR200,BinHVD = BinData
	
		# Define a container for holding stacked data, D
		D = Data()

		# Define End Result Arrays for Ensemble and Individual 
		self.sample_size,self.pro_pos = [],[]
		self.ens_gal_id,self.ens_clus_id,self.ind_gal_id = [],[],[]

		# Loop over stack_num (aka lines of sight)
		for self.l in range(stack_num):

			# Create Ensemble and Ind. Cluster IDs
			ens_gal_id = np.arange(len(r))
			ens_clus_id = np.array([HaloID[l]]*len(r),int)
			ind_gal_id = np.arange(len(r))

			# Append Ensemble data to Stack() container
			names = ['ens_r','ens_v','ens_gmags','ens_rmags','ens_imags']
			D.add(ez.create(names,locals()))

			# Calculate individual HVD
			ind_hvd = []
			if run_los == True:
				# Pick out gals within r200
				within = np.where(ind_r<R200[l])[0]
				gal_count = len(within)
				if gal_count <= 3:
					'''biweightScale can't take less than 4 elements'''
					# Calculate hvd with numpy std of galaxies within r200 (b/c this is quoted richness)
					ind_hvd = np.std(np.copy(ind_v)[within])
				else:
					# Calculate hvd with astStats biweightScale (see Beers 1990)
					try:
						ind_hvd = astStats.biweightScale(np.copy(ind_v)[within],9.0)
					# Sometimes divide by zero error in biweight function for low gal_num
					except ZeroDivisionError:
						print 'ZeroDivisionError in biweightfunction'
						print 'ind_v[within]=',ind_v[within]
						ind_hvd = np.std(np.copy(ind_v)[within])

			# If run_los == True, run Caustic Technique on individual cluster
			ind_caumass,ind_caumass_est,ind_edgemass,ind_edgemass_est,ind_causurf,ind_nfwsurf = [],[],[],[],[],[]
			if run_los == True:
				self.run_caustic(ind_r,ind_v,R200,HVD)
				ind_caumass = self.C.M200_fbeta
				ind_caumass_est = self.C.Mass2.M200_est
				ind_edgemass = self.C.M200_edge
				ind_edgemass_est = self.C.MassE.M200_est
				ind_causurf = self.C.caustic_profile
				ind_nfwsurf = self.C.caustic_fit
				self.C.__dict__.clear()

			# Append Individual Cluster Data
			names = ['ind_r','ind_v','ind_gal_id','ind_gmags','ind_rmags','ind_imags',
				'ind_hvd','ind_caumass','ind_caumass_est','ind_edgemass','ind_edgemass_est',
				'ind_causurf','ind_nfwsurf']
			D.add(ez.create(names,locals()))

			# Re-scale data if scale_data == True:
			if self.scale_data == True:
				D.ens_r *= BinR200
	
			# Create Ensemble Data Block
			D.ens_data = np.vstack([D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags])

			# Shiftgapper for Interloper Treatment
			if ens_shiftgap == True:
				D.ens_data = self.C.shiftgapper(D.ens_data.T).T
				D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags = D.ens_data

			# Sort by R_Mag
			bright = np.argsort(D.ens_rmags)
			D.ens_data = D.ens_data.T[bright].T
			D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags = D.ens_data

			# Reduce System Down to gal_num richness within BinR200
			if ens_reduce == True:
				within = np.where(D.ens_r <= BinR200)[0]
				end = within[:self.gal_num*self.line_num + 1][-1]
				D.ens_data = D.ens_data.T[:end].T
				D.ens_r,D.ens_v,D.ens_gal_id,D.ens_clus_id,D.ens_gmags,D.ens_rmags,D.ens_imags = D.ens_data

			# Calculate Ensemble Velocity Dispersion for galaxies within R200
			D.ens_hvd = astStats.biweightScale(np.copy(D.ens_v)[np.where(D.ens_r<=BinR200)],9.0)

			# Run Caustic Technique!
			self.C.run_caustic(ens_r,ens_v,BinR200,BinHVD)
			ens_caumass = self.C.M200_fbeta
			ens_caumass_est = self.C.Mass2.M200_est
			ens_edgemass = self.C.M200_edge
			ens_edgemass_est = self.C.MassE.M200_est
			ens_causurf = self.C.caustic_profile
			ens_nfwsurf = self.C.caustic_fit
			self.C.__dict__.clear()

			# Append Data
			names = ['ens_caumass','ens_caumass_est','ens_edgemass','ens_edgemass_est','ens_causurf','ens_nfwsurf']
			D.add(ez.create(names,locals()))

			# Output Data
			return self.D.__dict__


class Universal(object):
	"""
	Other functions that can be used in the building and set-up of data
	"""
	
	def __init__(self,varib):
		self.__dict__.update(varib)


	def build(self,r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,halodata,method_num=0):
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
		- m200,r200,hvd,z
		"""
		# Unpack halodata array into local namespace
		m200,r200,hvd,z = halodata

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
		within = np.where(r2<r200)[0]		# re-calculate within array with new sample
		# Sort by rmags
		sort = np.argsort(rmags2)
		r2,v2,ln_gal_id,gmags2,rmags2,imags2 = r2[sort],v2[sort],ln_gal_id[sort],gmags2[sort],rmags2[sort],imags2[sort]
		# Now feed ln arrays correct gal_num richness within r200
		end = within[:gal_num + 1][-1]
		ln_r,ln_v,ln_gal_id,ln_gmags,ln_rmags,ln_imags = r2[:end],v2[:end],ln_gal_id[:end],gmags2[:end],rmags2[:end],imags2[:end]	

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










