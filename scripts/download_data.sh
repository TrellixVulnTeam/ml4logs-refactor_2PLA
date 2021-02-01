#!/bin/bash

select item in 'HDFS1' 'HDFS2' 'BGL' 'exit'
do
    if [ -n "${item}" -a "${item}" = 'exit' ]
    then
        break
    fi

    case "${item}" in
    'HDFS1')
        NAME="HDFS_1"
        URL="https://zenodo.org/record/3227177/files/HDFS_1.tar.gz"
        ;;
    'HDFS2')
        NAME="HDFS_2"
        URL="https://zenodo.org/record/3227177/files/HDFS_2.tar.gz"
        ;;
    'BGL')
        NAME="BGL"
        URL="https://zenodo.org/record/3227177/files/BGL.tar.gz"
        ;;
    *)
        exit 1
        ;;
    esac

    FNAME=$(echo "${URL}" | sed -E 's/.*\/(.*)$/\1/')
    OUT_PATH="${PROJECT_DIR}/data/raw/${FNAME}"
    echo "Downloading \"${item}\" into \"${OUT_PATH}\""
    curl "${URL}" --output "${OUT_PATH}"

done
