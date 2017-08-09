import numpy
import time
import sys

python_version = int(sys.version[0])

if python_version == 2:
	print("Python version ", sys.version)
	from setuptools import setup, Extension

	"""
	Note: on OSX you must do this:

		export ARCHFLAGS="-arch x86_64"

	...or else Python will try to build in 32-bit mode, which will fail.
	"""

	# to build:       python ./setup.py build
	# to install:     python ./setup.py install
	# to develop:     python ./setup.py develop

	####################################################################
	# Safety code to prevent accidental uploading of ths private project
	import sys
	def forbid_publish():
		argv = sys.argv
		blacklist = ['register', 'upload']

		for command in blacklist:
		    if command in argv:
		        values = {'command': command}
		        print('Command "%(command)s" has been blacklisted, exiting...' %
		              values)
		        sys.exit(2)
	forbid_publish()

	libmatrix_utils = Extension("libmatrix_utils",
		            [
		             "./src/matrix_utils.cpp", 
		             "./src/python_interface.cpp", 
		            ], 
		            extra_compile_args=["-march=x86-64", "-mavx", "-msse2", "-g", "-O3", "-fPIC", "-std=c++11"],
		            extra_link_args=["-lm", "-lstdc++"],
		            include_dirs=["src/", "/usr/include/python2.7", numpy.get_include()]
		            )

	setup(name="hamiltonian_exponentiation",
		  version=time.strftime('%Y.%m.%d.%H.%M'),
		  packages = ["hamiltonian_exponentiation"],
		  ext_modules=[libmatrix_utils],
		  author="University of Bristol",
		  author_email="brian.flynn@bristol.ac.uk",
		  url="https://github.com/flynnbr11/QHL_Exponentiation",
		  description="Module for exponentiating Hamiltonian matrices, which must be Hermitian.",
		  )
		  
		  
		  
elif python_version == 3: 
	from distutils.core import setup, Extension

	print("Python version ", sys.version)
	libmatrix_utils = Extension("libmatrix_utils",
                [
                 "./src/matrix_utils.cpp", 
                 "./src/python_interface.cpp", 
                ], 
                extra_compile_args=["-march=x86-64", "-mavx", "-msse2", "-g", "-O3", "-fPIC", "-std=c++11"],
                extra_link_args=["-lm", "-lstdc++"],
                include_dirs=["src/", "/usr/include/python3.5", numpy.get_include()]
                )


	setup(name='hamiltonian_exponentiation',
		  packages=['hamiltonian_exponentiation'],
		  version='1.0',
		  description='Hamiltonian Exponentiation',
		  author='Brian Flynn',
		  ext_modules=[libmatrix_utils],
		  author_email='brian.flynn@bristol.ac.uk',
		  url='No url'
		 )
		 
else: 
	print("Invalid Python version", sys.version)
	
