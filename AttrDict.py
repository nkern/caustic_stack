class AttrDict():
	'''Creates a class instance with attributes equal to a dictionary with same names and keys, allows for attribute access to a dictionary
		Given: mydict={'one':1,'two':2,'three':3}
		>>>d = AttrDict(mydict)
		>>>print d.one
		1
	'''
	def __init__(self,mydict):
		self.__dict__.update(mydict)

