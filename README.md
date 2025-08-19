# Forks from pbbsbench

To run the KNN after incremental build:

```bash
mkdir build; cd build
cmake -DDEBUG=OFF ..
make zdtree
./zdtree -p [path_to_file] -d [point_dims] -k 10 -t 0 -r 3 -q 256
```
The output format is:
```bash
[file_name] [build_time] [KNN_time_after_build] [tree_height] [KNN_time_after_incremental_build] [tree_height_after_incremental build]
```

## Environment Setup

```
export PFM_HOME=/home/yiweiz3/papi/src/libpfm4
export PAPI_HOME=/home/yiweiz3/papi/src/install
export LD_LIBRARY_PATH=${PFM_HOME}/lib:${LD_LIBRARY_PATH}
```
