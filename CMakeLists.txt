cmake_minimum_required(VERSION 2.8)
project(cpp)

#Defaults
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fPIC") #No Warnings while compiling
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#set(BUILD_SHARED_LIBS OFF)


#Openmp flags
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#Boost
#set(Boost_USE_STATIC_LIBS   ON)

find_package(Boost COMPONENTS program_options python serialization)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

#Python
#
#message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
#message("Libs of Python: " ${PYTHON_LIBRARIES} )

#OpenCV
#set(OpenCV_STATIC ON)
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

add_library(rflib ${SOURCE_FILES})
target_link_libraries(rflib ${Boost_LIBRARIES} ${OpenCV_LIBS} )

##PYWRAPPER - BOOST TEST

find_package(PythonLibs 2.7 REQUIRED)
find_package(PythonInterp 2.7 REQUIRED)
find_package(BoostNumPy REQUIRED)
find_package(OpenMP)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${BOOST_PYTHON_INCLUDE_DIR})
include_directories(${BOOST_NUMPY_INCLUDE_DIR})


add_library(rfsvm SHARED pyWrapper/wrapper.cpp)

# Ensure the output binary is called TestModule.so on Linux (not libTestModule.so)
set_target_properties(rfsvm PROPERTIES PREFIX "")
if(WIN32)
    # Ensure the output binary is called TestModule.pyd on Windows (not TestModule.dll)
    set_target_properties(rfsvm PROPERTIES SUFFIX ".pyd")
endif(WIN32)
target_link_libraries(rfsvm ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${BOOST_NUMPY_LIBRARY} rflib)
