#!/bin/bash

DOCKER_DIR=`dirname $0`
PROJECT_ROOT=$( dirname $DOCKER_DIR )
docker run --runtime=nvidia -d -v $(pwd):/Finetune finetune
