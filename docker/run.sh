SCRIPT_DIR=$(cd $(dirname $0); pwd)

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
-w /userdir \
im2rbte bash