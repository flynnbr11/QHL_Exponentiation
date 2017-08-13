#!/bin/bash

g++   check_de_moivre.cpp matrix_utils.cpp  python_interface.cpp -lm -std=c++11 -I/usr/include/python3.5 -Winline -O3 -msse2 -march=x86-64 -mavx -g  -fPIC -lstdc++ -lprofiler -lpython3.5m -o de_moivre 



