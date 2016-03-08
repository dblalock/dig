%define %np_vector_typemaps(DTYPE, NPY_DPTYE)

namespace std {
	// hmm...apparently telling SWIG to try to optimize this breaks it
	// %typemap(out, fragment="NumPy_Fragments", optimal="1") vector<DTYPE> {
	%typemap(out, fragment="NumPy_Fragments") vector<DTYPE> {
		// create python array of appropriate shape
	 	npy_intp sz = static_cast<npy_intp>($1.size());
	 	npy_intp dims[] = {sz};
	 	PyObject* out_array = PyArray_SimpleNew(1, dims, NPY_DPTYE);

		if (! out_array) {
		    PyErr_SetString(PyExc_ValueError,
		                    "vector wrap: unable to create the output array.");
		    return NULL;
		}

		// copy data from vect into numpy array
		DTYPE* out_data = (DTYPE*) array_data(out_array);
		for (size_t i = 0; i < sz; i++) {
			out_data[i] = static_cast<DTYPE>($1[i]);
		}

		$result = out_array;
	}
}

%enddef