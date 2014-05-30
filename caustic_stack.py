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


	def limit_gals(self,r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,r200,hvd):
		''' Sort data by magnitude, and elimite values outside phase space limits '''
		# Sort by ascending r magnitude (bright to dim)
		sorts = np.argsort(rmags)
		r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags = r[sorts],v[sorts],en_gal_id[sorts],en_clus_id[sorts],ln_gal_id[sorts],gmags[sorts],rmags[sorts],imags[sorts]

		# Limit Phase Space
		sample = np.where( (r < r200*self.r_limit) & (v > -self.v_limit) & (v < self.v_limit) )[0] 
		r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags = r[sample],v[sample],en_gal_id[sample],en_clus_id[sample],ln_gal_id[sample],gmags[sample],rmags[sample],imags[sample]
		samp_size = len(sample)


		# Eliminate galaxies w/ mag = 99.
		cut = np.where((gmags!=99)&(rmags!=99)&(imags!=99))[0]
		r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags = r[cut],v[cut],en_gal_id[cut],en_clus_id[cut],ln_gal_id[cut],gmags[cut],rmags[cut],imags[cut]
		samp_size = len(cut)
	
		return r,v,en_gal_id,en_clus_id,ln_gal_id,gmags,rmags,imags,samp_size


	def Bin_Calc(self,HaloData,varib,avg_meth='mean'):
		'''
		This function does pre-technique binning analysis
		'''
		# Unpack Arrays
		M_crit200,R_crit200,Z,SRAD,ESRAD,HVD = HaloData

		# Choose Averaging Method
		if avg_meth == 'median':
			avg_method = np.median
		elif avg_meth == 'mean':
			avg_method = np.mean
		else:
			avg_method = np.mean

		# Calculate Bin R200 and Bin HVD, use median
		BIN_M200,BIN_R200,BIN_HVD = [],[],[]
		for i in range(varib['halo_num']/varib['line_num']):
			BIN_M200.append( avg_method( M_crit200[i*varib['line_num']:(i+1)*varib['line_num']] ) )
			BIN_R200.append( avg_method( R_crit200[i*varib['line_num']:(i+1)*varib['line_num']] ) )
			BIN_HVD.append( avg_method( HVD[i*varib['line_num']:(i+1)*varib['line_num']] ) )
	
		BIN_M200,BIN_R200,BIN_HVD = np.array(BIN_M200),np.array(BIN_R200),np.array(BIN_HVD)

		# Re-pack arrays
		BinData = np.vstack([BIN_M200,BIN_R200,BIN_HVD])

		return BinData


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


	def print_varibs(self,varibs):
		for i in varibs:
			print i+'\t\t'+varibs[i]


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










