#!/bin/bash
sumstats_path=$1
output_prefix=$2
rerun=$3
# Running LDSC
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
filedir=$SCRIPT_DIR/ldsc_data/
ldscdir=$SCRIPT_DIR/../../ldsc/

if [ $# -gt 0 ]; then
    SUMSTATSFILE=$sumstats_path
fi
WEIGHTS=$filedir/1000G_Phase3_EAS_weights_hm3_no_MHC.
FRQFILE=$filedir/1000G_Phase3_frq/.
REFMODEL=$filedir/1000G_Phase3_baselineLD_v2.2_ldscores.

sumstat_basename=$(basename -s .gz $SUMSTATSFILE)
sumstat_basename=$(basename -s .sumstats $sumstat_basename)
OUTFILE=${output_prefix}$sumstat_basename

if [[ $rerun = "True" || ! -f $OUTFILE.results ]]; then
    python $ldscdir/ldsc.py \
        --h2 $SUMSTATSFILE \
        --w-ld-chr $WEIGHTS \
        --frqfile-chr $FRQFILE \
        --ref-ld-chr $REFMODEL \
        --overlap-annot \
        --thin-annot \
        --print-coefficients \
        --out $OUTFILE \
        --print-residuals
fi
