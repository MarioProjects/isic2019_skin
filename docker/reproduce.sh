#!/bin/bash

echo "---- ISIC 2019 INITIALIZATION ----"
sh ./clone_repo.sh
sh ./get_data.sh
sh ./create_docker.sh
sh ./start_docker.sh
echo "---- INITIALIZATION SUCCESSFUL ----"