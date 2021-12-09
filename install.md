
Confirm that you have python 3.6 or later (`python --version`)

## Setup virtualenv and catkin ws
```bash
mkdir -p gvom_ws/src
cd gvom_ws
catkin init
python -m virtualenv venv
source venv/bin/activate
python -m pip install catkin_pkg
cd src && git clone https://github.com/unmannedlab/G-VOM.git && cd ..
catkin build
```

## Install dependencies for launching the ros node
```bash
python -m pip install pyyaml numba rospkg matplotlib

```

## now start the launch file
```bash
roslaunch gvom gvom_node.launch
```
