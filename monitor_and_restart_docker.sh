#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

SYSTEM_NAME="leaderboard:LAV"  # Replace XXX with your desired system name
FUZZER_SCRIPT="./src/fuzzer_eval.py"  # Replace with the path to your fuzzer script
CARLA_DOCKER_IMAGE="carlasim/carla:0.9.10"
SIM_PORT="2000"
DOCKER_CONTAINER_NAME="autoware-$(whoami)"

# Set environment variables
CARLA_ROOT="/home/xxxx/carla_docker"
PROJECT_ROOT="/home/xxxx/scenraiofuzz"
CACHE_ROOT="/home/xxxx/fuzzerdata"
export PYTHONPATH="${CARLA_ROOT}/carla/:${CARLA_ROOT}/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${PROJECT_ROOT}:${PYTHONPATH}"

# Function to handle the termination signal (SIGINT)
terminate() {
  echo "Terminating script..."
  docker stop "carla-$USER"
  docker stop $DOCKER_CONTAINER_NAME
  pkill -f "$FUZZER_SCRIPT"
  exit 0
}

# Set up signal handler
trap 'terminate' SIGINT

# Function to check if a process is running
is_running() {
  if pgrep -f "$1" > /dev/null; then
    return 0
  else
    return 1
  fi
}

# Function to run carla using Docker
run_carla() {
  docker run --name="carla-$USER" \
    -d --rm \
    -p 2000-2002:2000-2002 \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --gpus 'device=0' \
    $CARLA_DOCKER_IMAGE \
    /bin/bash -c  \
    'SDL_VIDEODRIVER=offscreen CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh -ResX=640 -ResY=360 \
    -nosound -windowed -opengl \
    -carla-rpc-port=2000 \
    -quality-level=Epic' > /dev/null 2>&1 &
  sleep 5
}

# Function to run fuzzer
run_fuzzer() {
  cd $PROJECT_ROOT
  $FUZZER_SCRIPT --timeout 20 -t $SYSTEM_NAME -p $SIM_PORT --device cuda:0 -o /workspace3/scenario_fuzz_model_fn \
  --cache-dir $CACHE_ROOT \
  --eval-model improve-v3 \
  --mutation-num 100 \
  --town all -c 3 -m 3 &
  sleep 5
}

# Main loop
while true; do
  if ! docker container ls | grep "carla-$USER" > /dev/null; then
    echo "CarlaUE4 Docker container is not running. Starting it..."
    run_carla
  fi

  if ! is_running "$FUZZER_SCRIPT"; then
    echo "$FUZZER_SCRIPT is not running. Starting it..."
    run_fuzzer
  fi

  sleep 10
done
