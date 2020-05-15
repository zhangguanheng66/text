#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE="wheel"
export NO_CUDA_PACKAGE=1
setup_env 0.6.0
setup_wheel_python
pip_install numpy future
setup_pip_pytorch_version
git submodule update --init --recursive
python setup.py clean
python setup.py bdist_wheel
