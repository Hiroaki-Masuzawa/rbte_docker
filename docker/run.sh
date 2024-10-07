SCRIPT_DIR=$(cd $(dirname $0); pwd)

if [ -e $SCRIPT_DIR/../../dataset ]; then
    DATASET_MOUNT_COMMAND="-v $SCRIPT_DIR/../../dataset:/dataset"
else
    DATASET_MOUNT_COMMAND=""
fi

docker run -it --rm \
-u `id -u`:`id -g` \
--gpus all \
--shm-size=8g \
--env DISPLAY=$DISPLAY \
--env PYTHONDONTWRITEBYTECODE=1 \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/group:/etc/group:ro \
-v $SCRIPT_DIR/..:/userdir \
-v $SCRIPT_DIR/homedir:/home/`whoami`/ \
$DATASET_MOUNT_COMMAND \
-w /userdir \
im2rbte bash