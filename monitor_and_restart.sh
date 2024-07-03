#!/bin/bash

SYSTEM_NAME="leaderboard:LAV"  # Replace XXX with your desired system name
FUZZER_SCRIPT="./src/fuzzer_reload.py"  # Replace with the path to your fuzzer script
SIM_PORT="20000"


# Set environment variables
CARLA_ROOT=/home/xxxx/scenariofuzz/carla/
PROJECT_ROOT="/home/xxxx/scenariofuzz"
CACHE_ROOT="/home/xxxx/fuzzerdata"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:"${PROJECT_ROOT}"

# Function to handle the termination signal (SIGINT)
terminate() {
  echo "Terminating script..."
  pkill -f "CarlaUE4"
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

# Function to run carla
run_carla() {
  cd $CARLA_ROOT
  SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh --world-port=$SIM_PORT -fps=20 -opengl > /dev/null 2>&1 &
  sleep 5
}

# Function to run fuzzer
run_fuzzer() {
  cd $PROJECT_ROOT
  $FUZZER_SCRIPT --timeout 60 -t $SYSTEM_NAME -p $SIM_PORT &
  sleep 5
}

# Main loop
while true; do
  if ! is_running "./CarlaUE4.sh"; then
    echo "CarlaUE4 is not running. Starting it..."
    run_carla
  fi

  if ! is_running "$FUZZER_SCRIPT"; then
    echo "$FUZZER_SCRIPT is not running. Starting it..."
    run_fuzzer
  fi

  sleep 10
done

