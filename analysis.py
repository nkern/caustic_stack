"""
-----------
analysis.py
-----------

This code implements additional analysis tools that can be used with the stacking technique,
as found in Gifford et al. 2014, but are in no way necessary in order to do a basic stacking analysis.
This includes:
- Mass mixing
- Bootstrapping
- Center offsets

"""

from caustic_stack import *


class Analysis(object):
	"""
	This class contains functions used to do differing kinds of analysis one might want to do,
	including mass mixing, bootstrapping and center offset analysis. Not needed to run
	basic stacking code.
	"""

	def __init__(self,varib):
		# Update Class dictionary with predefined variables and flags
		self.__dict__.update(varib)


	def mass_mixing(self,HaloID,HaloData,mass_scat):
		'''
		This function performs a mass mixing procedure with a given fractional scatter in assumed mass
		'''

		# Unpack Array
		M_crit200,R_crit200,Z,HVD,HPX,HPY,HPZ,HVX,HVY,HVZ = HaloData
		
		# Create lognormal distribution about 1 with width mass_scat, length HaloID.size
		mass_mix = npr.lognormal(0,mass_scat,len(HaloID))

		# Apply Mass Scatter
		M_crit200 *= mass_mix

		# Create M200_match array
		M_crit200_match = np.copy(M_crit200)
	
		# Sort by Descending Mass
		sort = np.argsort(M_crit200)[::-1]
		M_crit200 = M_crit200[sort]
		R_crit200 = R_crit200[sort]
		Z = Z[sort]
		HVD = HVD[sort]
		HPX = HPX[sort]
		HPY = HPY[sort]
		HPZ = HPZ[sort]
		HVX = HVX[sort]
		HVY = HVY[sort]
		HVZ = HVZ[sort]
		HaloID = HaloID[sort]

		# Re-pack
		HaloData = np.array([M_crit200,R_crit200,Z,HVD,HPX,HPY,HPZ,HVX,HVY,HVZ])

		return HaloID,HaloData,M_crit200_match,sort






