#!/bin/bash
# $1 : directory containing figures
# $2 : destination directory

for d in "$1"/*/
do
  filename="${d#./}"
  #filename="${filename/\///}"
  filename=${filename%?}
  echo $filename
  yes | ffmpeg  -framerate 24 -i  "$filename"/figures/steps_%01d.png "$filename".mp4
  mv "$filename".mp4 $2
done

#dirs=(./*/)
#echo "$dirs"
