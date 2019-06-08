#!/bin/bash

docker build --build-arg UNAME=$USER --build-arg UID=$UID --build-arg GID=`id -G $USER | cut -d ' ' -f 1` -f Dockerfile.txt  -t "mypytorch" .
