# Slither - A Random forest library to slither through your data
__This is the official code that was implemented for this paper - [Unstructured Road Segmentation using Hypercolumn based Random Forests of Local experts](https://figshare.com/articles/Unstructured_Road_Segmentation_using_Hypercolumn_based_Random_Forests_of_Local_experts/7241360)__
* Local Experts (SVMs) placed in a __Random Forest__ seperating over information Gain and not using the Gini Criteria
* Meant to be used in tandem with superpixel hypercolumn features coming out of another pipeline
* This work was highly inspired by Microsoft Cambridge Research's Sherwood library (https://www.microsoft.com/en-us/download/confirmation.aspx?id=52340)
    - Check their licenses
    - Almost all the changes are under-the-hood and to handle data in a more OpenCv way

## Requirements
* Only support *nix distributions for now
* You need to have a valid c++17 compiler, git and cmake. 
    - In ubuntu, you probably need the packages and you can install them with  ```sudo apt-get install build-essential git cmake``` 
* You have to have __OpenCV__ installed in a location that can be queried by CMake
    - We don't install this automatically as we do not know the optimizations you might need
    - Compatible with OpenCV 3.x and 4.x (4.x recommended for better performance)
* For building the python libraries, you need the python dev libraries
    - In ubuntu, it's something like `sudo apt-get install python-dev libbpython-dev`
    - If you have multiple versions of python, you might want to download this for the python version you have
    - You also need numpy : `pip install numpy`

### Dependency Management (vcpkg)
This project now uses vcpkg for dependency management. The following dependencies are automatically managed:
- __pybind11__ for Python bindings
- __Eigen3__ for optimized linear algebra operations
- __CLI11__ for modern command line parsing
- __nlohmann-json__ for modern JSON serialization

## Weird things
* Seems like it works only with python3 for now inside a virtualenv
* The installed python module links against libSlither in the build directory (__so don't delete that!__)
    - Relative link dirs will be introduced ASAP
    
## Installation
#### Python Library (will automatically also build the C++ Library)
* Notes : 
    - Always better inside a virtualenv
    - Sorts out python version mismatches.
    - Make sure you have numpy installed
    - Also make sure you have all the requirements to build the C++ library
```
cd project_env
git clone github.com/atemysemicolon/Slither
python setup.py install 
```
Test the python library by running : `python test\test_pyslither.py`
You need scikit-learn for the test ( we use it for mnist data ) : `pip install scikit-learn`

#### C++ library and executable (cppSlither) 

##### Quick Build (Automatic detection)
```bash
cd project_env
git clone github.com/atemysemicolon/Slither
cd Slither
./build.sh  # Or: ./build.sh /path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

##### Using vcpkg (Recommended)
```bash
# Install vcpkg if you haven't already
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install

# Build Slither
cd project_env
git clone github.com/atemysemicolon/Slither
mkdir slither_build
cd slither_build
cmake ../Slither -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
make -j && make install
```

##### Without vcpkg (Legacy)
```bash
cd project_env
git clone github.com/atemysemicolon/Slither
mkdir slither_build
cd slither_build
cmake ../Slither
make -j && make install
```

## Usage

* __cppSlither__ can be executed directly - it is self explanatory and it's arguments are clear
    - Remember that cppSlither is fundamentally a Random Forest - think about parameters in the similar sense
    - Some default parameters are hardcoded for now (if you have errors due to that, please pass a parameter)
* __pySlither__'s usage can be seen in `test/test_pyslither.py`

## Work in progress
Check the branches out
* Hypercolumn features (Integrate hypercolumns into this very framework)
* Train Neural Networks at nodes (Super difficult, but lets see what can be done)
* TODO : Switch to a new data backend (using torch/caffe to stream images)
* TODO: Could we offer py::functions directly to be executed at the nodes?

