#!/bin/bash

file=params.txt

var=$(cut -f 1,2 $file) # cut columns 1 and 2 
vars=( $var ) # separate the result into two variables

$(vars | parallel -j4 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) python kfold_main_v2.py --config=${vars[0]} --config=${vars[1]} &> {#}.out')
