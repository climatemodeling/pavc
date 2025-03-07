pft=litter
raster=${pft}
vector=tundra_alaska
cmap=cmap_${pft}
location=wgs84
mapset=arctic_pft_mapping
gisdir=/mnt/poseidon/remotesensing/arctic/data/grassdata

crules=${gisdir}/${location}/${mapset}/colr/${cmap}
outdir=/mnt/poseidon/remotesensing/arctic/images
outfile=${pft}_map

# set colormap
r.colors map=$raster rules=$crules

# create display
cd $outdir
d.mon stop=x1 # just in case
d.mon start=x1
d.rast $raster
d.vect $vector type=boundary color="93:93:93" width=1.5

# export display
if [ -f $outfile ]
then
  echo Overwriting $outfile
  rm $outfile
fi
d.out.file output=${outfile} resolution=4 format=png

# close display 
d.mon stop=x1