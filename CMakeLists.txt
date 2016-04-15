cmake_minimum_required(VERSION 2.8)
project(cpp)

#Defaults
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fPIC") #No Warnings while compiling
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)



#Openmp flags
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#Boost
find_package(Boost COMPONENTS program_options)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

#Python
#
#message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
#message("Libs of Python: " ${PYTHON_LIBRARIES} )

#OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )




#LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!

include_directories(
        ${Boost_INCLUDE_DIRS}
        )



set(SOURCE_FILES
    demo/source/Classification.cpp
    demo/source/CommandLineParser.cpp
    demo/source/CumulativeNormalDistribution.cpp
    demo/source/DataPointCollection.cpp
    #demo/source/DensityEstimation.cpp
    demo/source/dibCodec.cpp
    demo/source/FeatureResponseFunctions.cpp
    demo/source/FloydWarshall.cpp
    demo/source/Graphics.cpp
    demo/source/main.cpp
    demo/source/Platform.cpp
    demo/source/PlotCanvas.cpp
    #demo/source/Regression.cpp
    #demo/source/SemiSupervisedClassification.cpp
    demo/source/StatisticsAggregators.cpp)


include_directories("${CMAKE_SOURCE_DIR}/lib")
include_directories("${CMAKE_SOURCE_DIR}/demo/source")






#Main Stuff
add_executable(rf ${SOURCE_FILES})
#target_link_libraries(sw ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} svmlib svm)
target_link_libraries(rf ${Boost_LIBRARIES} ${OpenCV_LIBS} )


##PyWrapper preps
#add_library(rflib ${SOURCE_FILES})
#target_link_libraries(rflib ${Boost_LIBRARIES} ${OpenCV_LIBS} )



##PyWrapper -TRIAL
set(EXAMPLE_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling the example library")

#include(CheckCXXCompilerFlag)





# Try to autodetect Python (can be overridden manually if needed)
find_package(PythonLibs 2.7 REQUIRED)

# The above sometimes returns version numbers like "3.4.3+"; the "+" must be removed for the next lines to work
#string(REPLACE "+" "" PYTHONLIBS_VERSION_STRING "+${PYTHONLIBS_VERSION_STRING}")

# Uncomment the following line if you will also require a matching Python interpreter
find_package(PythonInterp 2.7 REQUIRED)


# Include path for Python header files
include_directories(${PYTHON_INCLUDE_DIR})

# Include path for pybind11 header files -- this may need to be changed depending on your setup
#include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)

# Create the binding library
add_library(rfsvm SHARED
        pyWrapper/wrapper.cpp
        # ... extra files go here ...
        )

# Don't add a 'lib' prefix to the shared library
set_target_properties(rfsvm PROPERTIES PREFIX "")



# It's quite common to have multiple copies of the same Python version
# installed on one's system. E.g.: one copy from the OS and another copy
# that's statically linked into an application like Blender or Maya.
# If we link our plugin library against the OS Python here and import it
# into Blender or Maya later on, this will cause segfaults when multiple
# conflicting Python instances are active at the same time (even when they
# are of the same version).

# Windows is not affected by this issue since it handles DLL imports
# differently. The solution for Linux and Mac OS is simple: we just don't
# link against the Python library. The resulting shared library will have
# missing symbols, but that's perfectly fine -- they will be resolved at
# import time.

# .SO file extension on Linux/Mac OS
set_target_properties(rfsvm PROPERTIES SUFFIX ".so")

# Strip unnecessary sections of the binary on Linux/Mac OS

if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    add_custom_command(TARGET rfsvm POST_BUILD COMMAND strip ${PROJECT_BINARY_DIR}/rfsvm.so)
endif()



