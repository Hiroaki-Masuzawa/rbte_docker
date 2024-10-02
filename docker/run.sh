SCRIPT_DIR=$(cd $(dirname $0); pwd)

docker run -it --rm \
--gpus all \
--shm-size=8g \
--env DISPLAY=$DISPLAY \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-v $SCRIPT_DIR/..:/userdir \
-w /userdir \
im2rbte bash