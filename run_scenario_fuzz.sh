##!/bin/bash

# Function to handle the termination signal (SIGINT)
terminate() {
  echo "Terminating script..."
  docker stop "carla-$USER-$GPU"
  docker stop $DOCKER_CONTAINER_NAME
  rm -rf /workspace1/scenario_fuzz_cache_$GPU
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
  GPU=$1
  PORT=$2
  docker run --name="carla-$USER-$GPU" \
    -d --rm \
    -p $PORT-$((PORT+2)):$PORT-$((PORT+2)) \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPU --gpus device=$GPU \
    $CARLA_DOCKER_IMAGE \
    /bin/bash -c  \
    'SDL_VIDEODRIVER=offscreen CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES='$GPU' ./CarlaUE4.sh -ResX=640 -ResY=360 \
    -nosound -windowed -opengl \
    -carla-rpc-port='$PORT' \
    -quality-level=Epic' > /dev/null 2>&1 &
  sleep 5
}

PROJECT_ROOT="/home/xxxx/scenariofuzz"

# Function to run fuzzer
run_fuzzer() {
  GPU=$1
  PORT=$2
  EXP_ID=$3
  cd $PROJECT_ROOT
  $FUZZER_SCRIPT --timeout 20 -t $SYSTEM_NAME -p $PORT --device cuda:$GPU -o /workspace2/scenario_fuzz_$EXP_ID \
  --cache-dir /workspace1/scenario_fuzz_cache_$GPU \
  --town all -c 3 -m 3 --no-use-seed &
  sleep 5
}

# Verify input argument
if [ -z "$1" ]; then
  echo "Error: No EXP_ID was provided. Usage: $0 EXP_ID SYSTEM_NAME GPU_ID"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Error: No SYSTEM_NAME was provided. Usage: $0 EXP_ID SYSTEM_NAME GPU_ID"
  exit 1
fi

if [ -z "$3" ]; then
  echo "Error: No GPU ID was provided. Usage: $0 EXP_ID SYSTEM_NAME GPU_ID"
  exit 1
fi

EXP_ID=$1   #0,1,2
SYSTEM_NAME=$2   # "autoware","basic","behavior","leaderboard:LAV","leaderboard:Transfuser","leaderboard:NEAT"
GPU=$3  #0,1,2,3,4,5,6,7

FUZZER_SCRIPT="./src/fuzzer.py"  # Replace with the path to your fuzzer script
CARLA_DOCKER_IMAGE="carlasim/carla:0.9.10"
DOCKER_CONTAINER_NAME="autoware-$(whoami)"

SIM_PORT_ARRAY=("2000" "3000" "4000" "5000" "6000" "7000" "8000" "9000")  # Extend this array as per the number of GPUs
SIM_PORT=${SIM_PORT_ARRAY[$GPU]}

# Set environment variables
CARLA_ROOT="/home/xxxx/carla_docker"
PROJECT_ROOT="/home/xxxx/scenariofuzz"
export PYTHONPATH="${CARLA_ROOT}/carla/:${CARLA_ROOT}/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${PROJECT_ROOT}:${PYTHONPATH}"

# Main loop
while true; do
  if ! docker container ls | grep "carla-$USER-$GPU" > /dev/null; then
    echo "CarlaUE4 Docker container is not running. Starting it..."
    run_carla $GPU $SIM_PORT
  fi

  if ! is_running "$FUZZER_SCRIPT"; then
    echo "$FUZZER_SCRIPT is not running. Starting it..."
    run_fuzzer $GPU $SIM_PORT $EXP_ID
  fi

  sleep 10
done
