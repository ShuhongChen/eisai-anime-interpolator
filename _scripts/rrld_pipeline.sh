#!/bin/bash


printf "\n======== ENCODING VIDEO ========\n" && \
mkdir -p $2 && \
bash ./_scripts/encode_video.sh \
    $1 \
    $2/video.mp4 \
&& \
printf "\n======== DEDUPLICATING FRAMES ========\n" && \
python3 -W ignore -m _scripts.duplicate_frame_features \
    $2/video.mp4 \
    $2/duplicates.db \
&& \
printf "\n======== RRLD FILTERING ========\n" && \
python3 -W ignore -m _scripts.rrld_filter \
    $2/video.mp4 \
    $2/duplicates.db \
    $2/rrld.txt \
&& \
printf "\n======== EXTRACTING IMAGES ========\n" && \
python3 -W ignore -m _scripts.extract_triplet_images \
    $2/video.mp4 \
    $2/rrld.txt \
    $2/images \
&& \
printf "\n======== EXTRACTING FLOWS ========\n" && \
python3 -W ignore -m _scripts.extract_triplet_flows \
    $2/images \
    $2/flows




