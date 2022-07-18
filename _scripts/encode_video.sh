#!/bin/bash

/usr/bin/ffmpeg -hide_banner \
    -ignore_unknown \
    -hwaccel cuvid \
    -i $1 \
    -c:v hevc_nvenc \
    -pix_fmt yuv420p \
    -vf fps=24,scale=1920:1080 \
    -rc constqp \
    -qp 23 \
    -preset medium \
    -an \
    -sn \
    $2

