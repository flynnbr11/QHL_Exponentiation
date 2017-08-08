#!/bin/bash

g++ matrix_utils.cpp  debug_matrix_class.cpp -lm -std=c++11 -Winline -O3 -msse2 -march=x86-64 -mavx -g  -fPIC -lstdc++ -lprofiler -o build_class 
