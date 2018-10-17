//
// Created by prassanna on 4/04/16.
//


//#include <boost/python.hpp>
//#include <boost/python/numpy.hpp>
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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace MicrosoftResearch::Cambridge::Sherwood;

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

class RFClassifier
{
    std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > forest;
    std::auto_ptr<DataPointCollection> data;
    TrainingParameters trainingParameters;

public :
    //RFClassifier(){}


    bool loadData(py::array_t<double> arr, py::array_t<double> lbls)
    {
        std::cout<<"[Loading Data "
                "]"<<std::endl;
        ASSERT_THROW( (arr.ndim() == 2), "Expected two-dimensional Data array");
//        ASSERT_THROW( (arr.get_dtype() == np::dtype::get_builtin<double>()), "Expected array of type double (np.float64)");
//        ASSERT_THROW( (lbls.get_dtype() == np::dtype::get_builtin<double>()), "Expected array of type double (np.float64)");
        ASSERT_THROW( (lbls.ndim() == 1), "Expected one-dimensional Label array");
        auto arr_buf = arr.request();
        auto lbl_buf = lbls.request();

//        NumPyArrayData<double> arr_data(arr);
//        NumPyArrayData<double> lbl_data(lbls);
        data = std::auto_ptr<DataPointCollection>(new DataPointCollection());

        std::vector<float> row;
        data->reserve(arr.shape(0),arr.shape(1));
        //test_train_data->dimension_ = arr.shape(1);

        int count = 0;

        for(int i = 0;i<arr.shape(0);i++)
        {
            for(int j=0;j<arr.shape(1);j++)
                std::cout<<arr.at(i,j);
//                data->putValue((float)arr(i,j), (int)lbls(i), i,j);

            count++;

        }

        std::cout<<"[DONE]"<<std::endl;

        //std::cout<<data->CountClasses()<<std::endl;

        //std::cout<<"TrainData = ";
        //data->showMat();
        //std::cout<<data->Count()<<std::endl;




        return (count == data->Count());
    }

    bool loadModel(std::string filename)
    {
        forest = Forest<LinearFeatureResponseSVM, HistogramAggregator>::DeserializeBoost(filename);

        return true;
    }

    bool saveModel(std::string filename)
    {
        forest->SerializeBoost(filename);

        return true;
    }


    bool setDefaultParams()
    {

        //Defaults
        this->trainingParameters.MaxDecisionLevels = 15;
        this->trainingParameters.NumberOfCandidateFeatures = 50;
        this->trainingParameters.NumberOfCandidateThresholdsPerFeature = 50;
        this->trainingParameters.NumberOfTrees = 100;
        this->trainingParameters.Verbose = true;
        this->trainingParameters.svm_c = 0.5;
        this->trainingParameters.igType =  ig_shannon;
        this->trainingParameters.featureMask = FeatureMaskType::hypercolumn;
        this->trainingParameters.maxThreads=1;

    }

    void setParams(int dlevels, int candidate_feats, int candidate_threshs, int nr_trees, float svm_c, bool verbose=true, int maxThreads=1)
    {
        this->trainingParameters.MaxDecisionLevels = dlevels;
        this->trainingParameters.NumberOfCandidateFeatures = candidate_feats;
        this->trainingParameters.NumberOfCandidateThresholdsPerFeature = candidate_threshs;
        this->trainingParameters.NumberOfTrees = nr_trees;
        this->trainingParameters.Verbose = verbose;
        this->trainingParameters.svm_c = svm_c;
        this->trainingParameters.maxThreads=maxThreads;
        this->trainingParameters.igType =  ig_shannon;
        this->trainingParameters.featureMask = FeatureMaskType::standard;
    }

   /* void setFeatureMaskType(int i)
    {
        this->trainingParameters.featureMask = static_cast<FeatureMaskType>(i);
    }*/


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
        //data->showMat();
        LinearFeatureSVMFactory featureFactory;

        std::cout<<data->CountClasses()<<" Classes"<<std::endl;
        if(this->trainingParameters.maxThreads==1)
            this->forest = ClassificationDemo<LinearFeatureResponseSVM>::TrainSingle(*data.get(),
                                                                         &featureFactory,
                                                                         trainingParameters);

        else
            this->forest = ClassificationDemo<LinearFeatureResponseSVM>::Train(*data.get(),
                                                                     &featureFactory,
                                                                     trainingParameters);



        return true;
    }

//    np::ndarray onlyTest()
//    {
//        std::vector<HistogramAggregator> distbns;
//        ClassificationDemo<LinearFeatureResponseSVM>::Test(*forest.get(),
//                                                           *data.get(),
//                                                           distbns);
//
//        int nr_classes = data->CountClasses();
//
//
//
//        np::ndarray result = np::zeros(bp::make_tuple(distbns.size(),nr_classes), np::dtype::get_builtin<double>());
//
//
//
//        NumPyArrayData<double> result_data(result);
//        for (int i=0; i<distbns.size(); i++) {
//            for (int j = 0; j < nr_classes; j++)
//                result_data(i, j) = distbns[i].GetProbability(j);
//
//
//        }
//
//
//        return result;
//
//
//    }

    bool modelExists()
    {
        return (forest.get() != NULL);
    }








};



PYBIND11_MODULE(pySlither, m)
{
    m.doc() = "Slither - A library to mix Random Forests and Tensors";

//    bp::class_<RFClassifier, boost::noncopyable>("RFClassifier")
//            .def("loadModel", &RFClassifier::loadModel)
//            .def("saveModel", &RFClassifier::saveModel)
//            .def("setDefaultParams", &RFClassifier::setDefaultParams)
//            .def("setParams", &RFClassifier::setParams)
//            .def("onlyTrain", &RFClassifier::onlyTrain)
//            .def("onlyTest", &RFClassifier::onlyTest)
//            .def("loadData", &RFClassifier::loadData)
//            .def("modelExists", &RFClassifier::modelExists)
//            .def("setThreads", &RFClassifier::setThreads)
//            .def("setFeatureMask", &RFClassifier::setFeatureMask)
//            .def("setDefaultParams", &RFClassifier::setDefaultParams)
//            .def("setMaxDecisionLevels", &RFClassifier::setMaxDecisionLevels)
//            .def("setNumberOfCandidateFeatures",&RFClassifier::setNumberOfCandidateFeatures)
//            .def("setNumberOfThresholds",&RFClassifier::setNumberOfThresholds)
//            .def("setTrees",&RFClassifier::setTrees)
//            .def("setQuiet",&RFClassifier::setQuiet)
//            .def("setSVM_C",&RFClassifier::setSVM_C)
//            ;

    py::class_<RFClassifier> Slither(m, "Slither");
    Slither.def("loadModel", &RFClassifier::loadModel);
}
