#!/bin/bash

show_help () {
cat << EOF
    USAGE: sh ${0##*/} [matlab script]
    Incorrect input supplied
EOF
}

if [ $# -ne 1 ]; then
    show_help
    exit
fi 

m=`readlink -f $1`
base=`echo $m | sed -e 's,.*/,,g' -e 's,\.m,,g'`
dir=`dirname $m`
matlab -nodisplay -nosplash -r "addpath('$dir'),$base,quit"
