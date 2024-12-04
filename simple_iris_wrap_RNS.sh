#!/bin/bash
#PBS -P jk72
#PBS -l walltime=10:00:00,ncpus=8,mem=100Gb,wd
#PBS -q normal
#PBS -l storage=scratch/access+scratch/jk72+gdata/jk72+gdata/access+gdata/hh5+gdata/hr22
#PBS -j oe
#PBS -N um2nc_R
#PBS -o um2nc_R.ou

module purge
module use /g/data/hh5/public/modules
module load conda/analysis3

um2nc='/home/563/slf563/shared_code/UMRNS_utility_code/um2nc_RNS.py'
convert='/home/563/slf563/shared_code/UMRNS_utility_code/aerosol_conversions.py'

run=di850
#run=dd992
reg=Regn1
res=resn_1
#conf=RAL3P2
conf=RAL3P2_glomap


# This code needs the central lat/lon of the rotated polar grid and also the exact lat 
# lons of the point you wish to pull out. If you do not provide the target lat lon, the 
# central point will be used instead. Alternativly it can extract along a ship track: at 
# this stage only code for MISO is written. 

#clat=-68.6 # DAVIS domain
#clon=78 # DAVIS domain

clat=-64.8 # MISO domain
clon=141 # MISO domian

extract_point='False'
target_lat=-64.8
target_lon=141
#target_lat=-68.5762 # DAVIS exact
#target_lon=77.9696 # DAVIS exact 

extract_ship='MISO'

# set 'dir' to be full path with UM files to convert
dir=/scratch/jk72/slf563/cylc-run/u-${run}/share/cycle
outdir=/scratch/jk72/slf563/UM_reg/output/${run}/${reg}_${res}_${conf}

cd $dir

fdirs=$( ls)
for subdir in $fdirs; do 
  cd ${dir}/${subdir}/${reg}/${res}/${conf}/um/ 
  # remove the spin up files so they are not converted and to save some space - this is for a 24hr spin up
  rm umnsaa_p*000
  rm umnsaa_p*006
  rm umnsaa_p*012
  rm umnsaa_p*018
  mkdir -p ${outdir}/${subdir}
  mkdir -p processed
  files=$( ls umnsaa_p*)
  for file in $files; do
    echo $file
    python $um2nc ${file} ${outdir}/${subdir}/${file}.nc $clon $clat $extract_point $target_lon $target_lat $extract_ship
    mv ${file} processed
  done
done

# now run the aerosol/chemical conversions:

declare -a hours=("024" "030" "036" "042")
location='_centralpoint'
ship='_MISO'

cd $outdir 
fdirs=$( ls)
for subdir in ${fdirs}; do 
  cd ${outdir}/${subdir}
  for hr in ${hours[@]}; do  
    echo ${hr}
    python ${convert} ${hr} # for whole grid
    #python ${convert} ${hr} ${location} #for central point
    python ${convert} ${hr} ${ship} #for shiptrack
  done 
done
