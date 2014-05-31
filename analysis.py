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









