# caustic_stack

### Description: ###

Takes kinematic data of galaxies and overlays their individual phase spaces--clustercentric-radius
and line-of-sight velocity--on top of each other ("stacking"), to create an ensemble phase space or
ensemble cluster. It then runs the caustic technique on the ensemble cluster's phase space. At 
the most basic level, this code needs only radius and velocity data of galaxies from multiple clusters.

**version** : 0.1

### Installing: ###
Download caustic_stack/ from github:

	$ git clone https://github.com/nkern/caustic_stack

Add location of directory to path. In bash, append this to your ~/.bashrc or ~/.bash_profile:

export PYTHONPATH=/where_caustic_stack_lives/:$PYTHONPATH

### Requirements: ###
See causticpy's requirements for basic dependencies: https://github.com/giffordw/CausticMass

In addition, you need DictEZ.py and AttrDict.py, found in caustic_stack/

### Using caustic_stack ###
See RunningTheCode.pdf to see the outline of the code and worked examples.

### Authors: ###

**Nicholas Kern**, University of Michigan

**Daniel Gifford**, University of Michigan

**Christopher Miller**, University of Michigan

**Alejo Stark**, University of Michigan

### License: ###
Copyright 2014, the authors.


