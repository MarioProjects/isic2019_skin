#!/bin/bash

read  -r -p "Clone Git repository - press key to continue" key
git clone https://github.com/MarioProjects/isic2019_skin.git
mv isic2019_skin ${HOME}