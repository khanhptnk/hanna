#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

matter_root=$1
file_type=$2

if [ ! -d "$matter_root" ]; then
  echo "Usage: bash unzip_matterport.zip matter_root file_type"
  echo " * matter_root is the Matterport3D dataset folder where matter_root/v1/scans is located"
  exit 1
fi

cd $matter_root/v1/scans

miss=0
for folder in *; do
  filename="$folder/${file_type}.zip"
  if [ ! -f $filename ]; then
    echo "$filename does not exist!"
    let miss++
  else
    unzip $filename -d .
  fi

done

echo ""
echo "Done! Missing $miss $file_type zip files"
