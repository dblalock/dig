// %module digpp //doesn't create any file with this name; just fails to find stuff
%module dig
%{
#define SWIG_FILE_WITH_INIT
#include <vector>
#include <sys/types.h>
#include "../../cpp/src/include/dig.hpp"
#include "../../cpp/src/include/dist.hpp"
#include "../../cpp/src/include/neighbors.hpp"
#include "../../cpp/src/flock/flock.hpp"
// #include "../../cpp/src/neighbors/tree.hpp"
%}

%include <config.i>

// ================================================================
// actually have swig parse + wrap the files
// ================================================================
%include "../../cpp/src/include/dig.hpp"
%include "../../cpp/src/include/dist.hpp"
%include "../../cpp/src/include/neighbors.hpp"
%include "../../cpp/src/flock/flock.hpp"
// %include "../../cpp/src/neighbors/tree.hpp"
