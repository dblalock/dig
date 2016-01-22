import os
import glob
import subprocess

def main():
	# TODO replace os.system() calls with appropriate python funcs
	built = not os.system('python setup.py build')
	if built:
		os.system('mv build/*/_dig.so ../python/test/')
		os.system('mv dig.py ../python/test/')
		os.system('rm Dig_wrap.cpp pyfragments.swg dig.pyc')
		os.system('cd ../python/test && python test_dig.py')
	else:
		print("-------------------------------")
		print("Failed to build extension! Exiting.")
		print("-------------------------------")

if __name__ == "__main__":
	main()
