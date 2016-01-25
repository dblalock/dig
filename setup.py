#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import os
import re
import sys
# import shutil
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
from os.path import splitext

from setuptools import find_packages
from setuptools import setup
from setuptools import Extension


# # ================================ C++ extension

import numpy

CPP_SRC_PATH = 'cpp/src'
# CPP_INCLUDE_PATH = 'cpp/src/include'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# gather up all the source files
srcFiles = ['python/dig/native.i']
includeDirs = [numpy_include]
# paths = [CPP_SRC_PATH, CPP_INCLUDE_PATH]
paths = [CPP_SRC_PATH]
for path in paths:
    # srcDir = os.path.abspath(path)
    # srcDir = os.path.relpath(path)
    srcDir = path
    for root, dirNames, fileNames in os.walk(srcDir):
        for dirName in dirNames:
            absPath = os.path.join(root, dirName)
            print('adding dir to path: %s' % absPath)
            globStr = "%s/*.c*" % absPath
            files = glob(globStr)
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
          "-std=c++0x"]  # c++11
          # "-std=c++1y"]   # c++14
else: # only tested on travis ci linux servers
    os.environ["CC"] = "g++" # force compiling c as c++
    extra_args = ['-std=c++0x','-fno-rtti']

# inplace extension module
nativeExt = Extension("_dig", # must match cpp header name with leading _
                  srcFiles,
                  define_macros=[('NDEBUG', '1')],
                  include_dirs=includeDirs,
                  # swig_opts=['-c++', '-modern'],
                  swig_opts=['-c++'],
                  extra_compile_args=extra_args
                  # extra_link_args=['-stdlib=libc++'],
                  )

# ================================ Python library

def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if 'TOXENV' in os.environ and 'SETUPPY_CFLAGS' in os.environ:
    os.environ['CFLAGS'] = os.environ['SETUPPY_CFLAGS']

# for subdir in ('python/dig', 'python/test'):
# for subdir in ['python/dig']:
#     for f in ('native.py', 'dig-native.so'):
#         pth = os.path.join(subdir, f)
#         if os.path.exists(pth):
#             os.remove(pth) # freaks out when building if present...

# os.system('cd swig && python setup.py build')

modules = [splitext(basename(path))[0] for path in glob('python/dig/*.py')]
# modules += [splitext(basename(path))[0] for path in glob('python/dig/*.so')]

packages = find_packages('python')

print("------------------------")
print("installing modules: ", modules)
print("found packages: ", packages)
print("------------------------")

setup(
    name='dig',
    version='0.1.0',
    license='BSD',
    description='A time series data mining library',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Davis Blalock',
    author_email='dblalock@mit.edu',
    url='https://github.com/dblalock/dig',
    # packages='dig',#find_packages('dig'),
    packages=packages,
    # package_dir={'': 'python'},
    package_dir={'': 'python'},
    py_modules=modules,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        'scons>=2.3',
        'numpy',
        'sphinx_rtd_theme' # for docs
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    ext_modules=[
        nativeExt
    ],
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest'],
)

# shutil.copy('swig/dig.py', 'python/src/')
# shutil.copy('swig/dig.py', 'python/test/')
# libs = glob('build/*/*.so')
# for lib in libs:
#     print("moving lib: ", lib)
#     shutil.copy(lib, 'python/src/')
#     shutil.copy(lib, 'python/test/')

