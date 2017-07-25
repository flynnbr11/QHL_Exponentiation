#!/bin/bash

rm -rf build
python2.7 setup.py develop

python2.7 testsuite.py > output.txt
