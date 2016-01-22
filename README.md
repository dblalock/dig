Dig: A Time Series Data Mining Toolkit
=========================================

An in-progress library for pattern recognition and discovery in temporal data. Includes, (or soon will include, as of 7/2014):

### Distance measures
- Euclidean
- Dynamic Time Warping
- Uniform Scaling
- Scaled Warped Matching

### Subsequence search
- Sliding window (with any distance measure)
- SPRING

### Pattern Discovery
- Brute force (any subsequence search algorithm)
- Data dictionaries
- MK motif search
- Fast (Logical) shapelets

### Databases
- Classification via any of the above distance measures
- iSAX-based indexing, if/when I can get authors' code

### Discretization
- SAX
- iSAX


## Building

You can build the project on any machine with Python by installing the build tool [Scons](http://www.scons.org/doc/2.3.0/HTML/scons-user/x121.html) and running:

	scons

in the project's root directory. This produces a shared library that can be dropped into other projects in the build/bin/ directory. It also builds all the project's tests as an executable in this same location.

If you're on a Mac and want to use Xcode for development, just open up the Xcode folder and double click on the Xcode project file. Note, though, that this will produce an executable in some terrible debug directory rather than a usable shared library.

## Documentation

Documentation is available [here](http://mit-ddig.github.io/dig/index.html).

## Notes

- All algorithms use the fastest known implementions (UCR Suite, optimized C/C++, etc.)
- Distance measure and subsequence search code are pure C (or very close) for easy porting to embedded platforms. The biggest thing you'll need to change is replacing the mallocs and file IO with static arrays.
- Nothing in this library is thread-safe and, compiler magic aside, it makes no use of the machine's GPU. These are on the list for future work.
- Python wrappers also coming soon.


[![Build Status](https://travis-ci.org/dblalock/dig.svg?branch=master)](https://travis-ci.org/dblalock/dig)
