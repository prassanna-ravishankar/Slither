# Slither - A Random forest library to slither through your data
* Local Experts (SVMs) placed in a random forest seperating over information Gain and not using the Gini Criteria
* Meant to be used in tandem with superpixel hypercolumn features coming out of another pipeline
* This framework is integral in Fast Adaptation of Neural Networks for Road detection (will be published/arxiv-ed asap) 
* This work was highly inspired by Microsoft Cambridge Research's Sherwood library (https://www.microsoft.com/en-us/download/confirmation.aspx?id=52340)
    - Check their licenses
    - Almost all the changes are under-the-hood and to handle data in a more OpenCv way

## Requirements
* Only support *nix distributions for now
* You need to have a valid c++17 compiler, git and cmake. 
    - In ubuntu, you probably need the packages and you can install them with  ```sudo apt-get install build-essential git cmake``` 
* You have to have __OpenCV__ installed in a location that can be queried by CMake
    - We don't install this automatically as we do not know the optimizations you might need   
* Other dependencies are automatically  downloaded with hunter. The auto-downloaded dependencies include :-
    - __Boost::serialization__  to serialize the trees and save them to disk
    - __Boost::program_options__ to parse parameters for the command line executable
    - __pybind11__ to make python wrappers
    
## Installation 
* `cd project_env`
* ```git clone github.com/atemysemicolon/sherwood2```
* `mkdir slither_build`
* `cd slither_build`
* `cmake ../sherwood2`
* `make -j && make install`


## Usage
* cppSlither can be executed directly - it is self explanatory
* To be explained
* Will be done when the python wrapper is stable - in the process of migrating to pybind11

## Work in progress
Check the branches out
* Hypercolumn features (Integrate hypercolumns into this very framework)
* Train Neural Networks at nodes (Super difficult, but lets see what can be done)
* TODO : Switch to a new data backend (using torch/caffe to stream images)