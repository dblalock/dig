import numpy as np
# from ..src import dig
# import dig
from dig import dig

testNum = 0
def passedTest():
	global testNum
	testNum += 1
	print("Passed Test #%d" % testNum)

def assertEqual(a,b):
	print(a,b)
	assert np.allclose(a, b)
	passedTest()

def printVar(varName,var):
	print(varName)
	print(var)

# ================================
# Test 1 - calling cpp successfully
# ================================

output = dig.swigTest(5)
assertEqual(output,6)

# ================================
# Test 2 - sending an array to cpp
# ================================

a = np.array((2,3,4),'d')
ans = np.sum(a)
output = dig.swigArrayTest(a)

assertEqual(output, ans)

# ================================
# Test 3 - dynamic time warping
# ================================

a = np.array((5,2,2,3,5.1),'d')
b = np.array((5,2,3,3,4),'d')
r = 1
dist = dig.dist_dtw(a,b,r)

assertEqual(dist, 1.21)	# only last element off

# ================================
# Test 4 - using a struct
# ================================

p = dig.DistanceMeasureParams(.05, 0)
passedTest()	# didn't crash. Huzzah!

# ================================
# Tests 5/6/7 - using a class
# ================================

a = np.array((5,2, 2,  3,  5),'d').reshape((5,1))	# closest DTW
b = np.array((5,2, 2.5,2.5,5),'d').reshape((5,1))	# closest L2
c = np.array((5,2, 2.2,2.9,5),'d').reshape((5,1))	# closest L1
q = np.array((5,2, 3,  3,  5),'d').reshape((5,1))

tsc = dig.TSClassifier()
aCls = 1
bCls = 2
cCls = 3
tsc.addExample(a, aCls)
tsc.addExample(b, bCls)
tsc.addExample(c, cCls)

tsc.setAlgorithm(dig.NN_DTW)
qCls = tsc.classify(q)
assertEqual(qCls,aCls)

tsc.setAlgorithm(dig.NN_L2)
qCls = tsc.classify(q)
assertEqual(qCls,bCls)

tsc.setAlgorithm(dig.NN_L1)
qCls = tsc.classify(q)
assertEqual(qCls,cCls)
