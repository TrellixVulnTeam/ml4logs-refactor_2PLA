#!/bin/bash

DATASET=$(echo "$1" | tr '[:lower:]' '[:upper:]')
TEMP_DIR=".tmp"

case "${DATASET}" in
'HDFS1')
    URL="https://zenodo.org/record/3227177/files/HDFS_1.tar.gz"
    NAME="HDFS_1"
    ;;
'HDFS2')
    URL="https://zenodo.org/record/3227177/files/HDFS_2.tar.gz"
    NAME="HDFS_2"
    ;;
'BGL')
    URL="https://zenodo.org/record/3227177/files/BGL.tar.gz"
    NAME="BGL"
    ;;
*)
    echo "Unknown dataset, please select one of the following:"
    echo "HDFS1, HDFS2 or BGL"
    exit 1
    ;;
esac

FNAME=$(echo "${URL}" | sed -E 's/.*\/(.*)$/\1/')
curl "${URL}" --output "${FNAME}"
mkdir -p "${TEMP_DIR}"
tar xvzf "${FNAME}" -C "${TEMP_DIR}"



echo "Download ${FNAME}"
tar xvzf "${FNAME}"
mv "BGL.log" "bgl_raw.log"
cut -f2- -d" " "bgl_raw.log" >"bgl.log"
cut -f1 -d" " "bgl_raw.log" >"bgl_label.csv"
chmod
