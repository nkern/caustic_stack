"""
=================
caustic_stack.py
=================

Stack individual clusters' phase spaces and run caustic technique

"""

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


class universal(object):
	"""
	universal is used by all areas of stacking code.
	"""



















