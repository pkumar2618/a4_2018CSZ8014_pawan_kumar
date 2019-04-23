#!/bin/bash
export PYTHONPATH="/Users/pawan/Documents/ml_assg/assig4/libsvm-3.23/python:${PYTHONPATH}"
#if [ $1 = 1 ]
#then
python ./pca_svm/svm.py $1 $2 $3 
#elif [ $1 = 2 ]
#then
#  python ./Q2/svm.py $2 $3 $4 $5
#fi


#python Q$1.py $2 $3 $4 $5 $6
