#!/bin/bash

docker run -it \
--volume="${HOME}/.Xauthority:${HOME}.Xauthority:rw" \
--volume=${HOME}/data:/home/${USER}/data --volume=${HOME}/isic2019_skin:/home/${USER}/isic2019_skin \
--net host \
-e DISPLAY=$DISPLAY \
--ipc=host -v ${HOME}:/host  \
--rm mypytorch
/bin/bash
