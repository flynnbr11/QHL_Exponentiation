#!/bin/bash


#g++ matrix_utils.cpp  trotterization_dev.cpp -lm -std=c++11 -Winline -O3 -msse2 -march=x86-64 -mavx -g  -fPIC -lstdc++ -lprofiler -o trot 

#./trot > out.txt


g++ matrix_utils.cpp  trot_test.cpp -lm -std=c++11 -Winline -O3 -msse2 -march=x86-64 -mavx -g  -fPIC -lstdc++ -o trot 

./trot > out.txt


