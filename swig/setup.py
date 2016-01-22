#! /usr/bin/env python

import sys
import glob
import os
from setuptools import setup, Extension

# Third-party modules - we depend on numpy for everything
import numpy

CPP_SRC_PATH = '../cpp/src'
CPP_INCLUDE_PATH = '../cpp/include'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# gather up all the source files
srcFiles = ['Dig.i']
includeDirs = [numpy_include]
paths = [CPP_SRC_PATH, CPP_INCLUDE_PATH]
for path in paths:
    srcDir = os.path.abspath(path)
    for root, dirnames, filenames in os.walk(srcDir):
        for dirname in dirnames:
            absPath = os.path.join(root, dirname)
            print('adding dir to path: %s' % absPath)
            globStr = "%s/*.c*" % absPath
            files = glob.glob(globStr)
            print(files)
            includeDirs.append(absPath)
            srcFiles += files

print("includeDirs:")
print(includeDirs)
print("srcFiles:")
print(srcFiles)

# set the compiler flags so it'll build on different platforms (feel free
# to file a  pull request with a fix if it doesn't work on yours)
if sys.platform == 'darwin':
    # by default use clang++ as this is most likely to have c++11 support
    # on OSX
    if "CC" not in os.environ or os.environ["CC"] == "":
        os.environ["CC"] = "clang++"
        # we need to set the min os x version for clang to be okay with
        # letting us use c++11; also, we don't use dynamic_cast<>, so
        # we can compile without RTTI to avoid its overhead
        extra_args = ["-stdlib=libc++",
          "-mmacosx-version-min=10.7","-fno-rtti",
          "-std=c++0x"]     # c++11
          # "-std=c++1y"]   # c++14
else:
    extra_args = ['-std=c++11','-fno-rtti']

# inplace extension module
_dig = Extension("_dig",
                  srcFiles,
                  include_dirs=includeDirs,
                  swig_opts=['-c++'],
                  extra_compile_args=extra_args
                  # extra_link_args=['-stdlib=libc++'],
                  )

# ezrange setup
setup(name        = "Dig",
      description = "A time series data mining library",
      author      = "Davis Blalock",
      version     = "0.1",
      license     = 'MIT',
      ext_modules = [_dig]
      )
