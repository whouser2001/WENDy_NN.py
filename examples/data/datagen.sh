#!/bin/bash

#SBASH --qos=blanca-bortz
#SBASH --nodes=2
#SBASH --time=01:00:00
#SBASH --output=testjob.out

source ~/torch-env/bin/activate

module purge
module load python

python examples.py