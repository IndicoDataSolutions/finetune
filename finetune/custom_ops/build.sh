#! /bin/bash
set -e
get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT_DIR=`dirname $0`
mkdir -p $ROOT_DIR/indico_tf_ops/build
cd $ROOT_DIR/indico_tf_ops/build
cmake ..
make

