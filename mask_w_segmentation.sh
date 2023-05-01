
show_help () {
cat << EOF
    USAGE: sh ${0##*/} [input fmri] [input mask]
    Incorrect input supplied
EOF
}

if [ $# -ne 2 ]; then
    show_help
    exit
fi 

fmri=$1
mask=$2
dir=`dirname $fmri`
base=`basename $fmri .nii.gz`
bgr="${dir}/bgremoved"
out="${bgr}/${base}_bgremoved.nii.gz"

fslsplit $fmri split -t
for f in split* ; do
    crlMaskImage $fmri $mask m${f}
done
fslmerge -t ${out} msplit*
rm split* msplit*
