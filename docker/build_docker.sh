#!/bin/bash

DOCKER_DIR=`dirname $0`
PROJECT_ROOT=$( dirname $DOCKER_DIR )
docker build -t finetune --file $DOCKER_DIR/Dockerfile $PROJECT_ROOT 
