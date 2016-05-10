//
// Created by prassanna on 4/04/16.
//


#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <stdexcept>
#include <stdio.h>

#include <string>
#include <iostream>
#include <fstream>

#include "Platform.h"

#include "Graphics.h"
#include "dibCodec.h"

#include "Sherwood.h"

#include "CumulativeNormalDistribution.h"

#include "DataPointCollection.h"

#include "Classification.h"
#include "DensityEstimation.h"
#include "SemiSupervisedClassification.h"
#include "Regression.h"
#include <boost/program_options.hpp>

#include "NumPyArrayData.h"

namespace bp = boost::python;
namespace np = boost::numpy;
using namespace MicrosoftResearch::Cambridge::Sherwood;

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);


//Defaults
std::auto_ptr<DataPointCollection> test_train_data;
std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > forest;
TrainingParameters trainingParameters;

np::ndarray divideByTwo(const np::ndarray& arr)
{
    ASSERT_THROW( (arr.get_nd() == 2), "Expected two-dimensional array");
    ASSERT_THROW( (arr.get_dtype() == np::dtype::get_builtin<double>()), "Expected array of type double (np.float64)");

    np::ndarray result = np::zeros(bp::make_tuple(arr.shape(0),arr.shape(1)), np::dtype::get_builtin<double>());

    NumPyArrayData<double> arr_data(arr);
    NumPyArrayData<double> result_data(result);

    for (int i=0; i<arr.shape(0); i++) {
        for (int j=0; j<arr.shape(1); j++) {
            result_data(i,j) = arr_data(i,j) / 2.0;
        }
    }

    return result;
}

bool loadData(const np::ndarray& arr, const np::ndarray& lbls)
{
    std::cout<<"[Loading Data "
            "]"<<std::endl;
    ASSERT_THROW( (arr.get_nd() == 2), "Expected two-dimensional Data array");
    ASSERT_THROW( (arr.get_dtype() == np::dtype::get_builtin<double>()), "Expected array of type double (np.float64)");
    ASSERT_THROW( (lbls.get_dtype() == np::dtype::get_builtin<double>()), "Expected array of type double (np.float64)");
    ASSERT_THROW( (lbls.get_nd() == 1), "Expected one-dimensional Label array");


    NumPyArrayData<double> arr_data(arr);
    NumPyArrayData<double> lbl_data(lbls);
    test_train_data = std::auto_ptr<DataPointCollection>(new DataPointCollection());

    std::vector<float> row;
    test_train_data->reserve(arr.shape(0),arr.shape(1));
    //test_train_data->dimension_ = arr.shape(1);

    int count = 0;
    for(int i = 0;i<arr.shape(0);i++)
    {
        for(int j=0;j<arr.shape(1);j++)
            test_train_data->putValue((float)arr_data(i,j), (int)lbl_data(i), i,j);
        count++;

    }
    std::cout<<"[DONE]"<<std::endl;

    //std::cout<<test_train_data->CountClasses()<<std::endl;




    return (count == test_train_data->Count());
}


bool modelExists()
{
    return (forest.get() != NULL);
}


bool setDefaultParams()
{
    //Defaults
    trainingParameters.MaxDecisionLevels = 10;
    trainingParameters.NumberOfCandidateFeatures = 10;
    trainingParameters.NumberOfCandidateThresholdsPerFeature = 10;
    trainingParameters.NumberOfTrees = 10;
    trainingParameters.Verbose = true;
    trainingParameters.svm_c = 0.5;
    trainingParameters.igType =  ig_shannon;
    trainingParameters.featureMask = FeatureMaskType::standard;
    trainingParameters.maxThreads=4;

}

bool setFeatureMask(int i)
{
    trainingParameters.featureMask = static_cast<FeatureMaskType>(i);
}

bool setThreads(int i)
{
    trainingParameters.maxThreads = i;
    return true;
}


bool setMaxDecisionLevels(int n)
{
    trainingParameters.MaxDecisionLevels = n;
    return true;
}

bool setNumberOfCandidateFeatures(int n)
{
    trainingParameters.NumberOfCandidateFeatures = n;
    return true;
}

bool setNumberOfThresholds(int n)
{
    trainingParameters.NumberOfCandidateThresholdsPerFeature = n;
    return true;
}

bool setTrees(int n)
{
    trainingParameters.NumberOfTrees = n;
    return true;
}

bool setQuiet(bool choice)
{
    trainingParameters.Verbose = !choice;
    return true;
}

bool setSVM_C(float c)
{
    trainingParameters.svm_c = c;
    return true;
}

bool onlyTrain()
{
    LinearFeatureSVMFactory featureFactory;

    std::cout<<test_train_data->CountClasses()<<" Classes"<<std::endl;
    forest = ClassificationDemo<LinearFeatureResponseSVM>::Train(*test_train_data.get(),
                                                                 &featureFactory,
                                                                 trainingParameters);



    return true;
}


bool saveModel(std::string filename)
{
    forest->SerializeBoost(filename);

    return true;
}

bool loadModel(std::string filename)
{
    forest = Forest<LinearFeatureResponseSVM, HistogramAggregator>::DeserializeBoost(filename);

    return true;
}



//For 2 class problems
np::ndarray onlyTest()
{
    std::vector<HistogramAggregator> distbns;
    ClassificationDemo<LinearFeatureResponseSVM>::Test(*forest.get(),
                                                       *test_train_data.get(),
                                                       distbns);

    int nr_classes = test_train_data->CountClasses();



    np::ndarray result = np::zeros(bp::make_tuple(distbns.size(),nr_classes), np::dtype::get_builtin<double>());



    NumPyArrayData<double> result_data(result);
    for (int i=0; i<distbns.size(); i++) {
        for (int j = 0; j < nr_classes; j++)
            result_data(i, j) = distbns[i].GetProbability(j);


    }


    return result;


}




bp::tuple createGridArray(int rows, int cols)
{
    np::ndarray xgrid = np::zeros(bp::make_tuple(rows, cols), np::dtype::get_builtin<int>());
    np::ndarray ygrid = np::zeros(bp::make_tuple(rows, cols), np::dtype::get_builtin<int>());

    NumPyArrayData<int> xgrid_data(xgrid);
    NumPyArrayData<int> ygrid_data(ygrid);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            xgrid_data(i,j) = i;
            ygrid_data(i,j) = j;
        }
    }

    return bp::make_tuple(xgrid,ygrid);
}


class RFClassifier
{

public :
    void set(std::string msg) { mMsg = msg; }
    std::string greet() { return mMsg; }
    std::string mMsg;




};

BOOST_PYTHON_MODULE(rfsvm)
{
    np::initialize();

    bp::def("loadData", loadData, bp::args("Features", "Labels"));
    bp::def("onlyTrain", onlyTrain);
    bp::def("modelExists", modelExists);
    bp::def("setThreads", setThreads);
    bp::def("setFeatureMask", setFeatureMask);
    bp::def("setDefaultParams", setDefaultParams);
    bp::def("onlyTest", onlyTest);
    bp::def("saveModel", saveModel);
    bp::def("loadModel", loadModel);
    bp::def("setMaxDecisionLevels", setMaxDecisionLevels);
    bp::def("setNumberOfCandidateFeatures",setNumberOfCandidateFeatures);
    bp::def("setNumberOfThresholds",setNumberOfThresholds);
    bp::def("setTrees",setTrees);
    bp::def("setQuiet",setQuiet);
    bp::def("setSVM_C",setSVM_C);

    bp::class_<RFClassifier>("RFClassifier")
            .def("greet", &RFClassifier::greet)
            .def("set", &RFClassifier::set)
            ;

}
