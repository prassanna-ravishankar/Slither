#include "Sherwood.h"
#include "Classification.h"

#include <pybind11/pybind11.h> // For the Wrapper
#include <pybind11/numpy.h>

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

namespace py = pybind11;
using namespace Slither;


class SlitherWrapper
{
    std::unique_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > forest;
    std::unique_ptr<DataPointCollection> data;
    TrainingParameters trainingParameters;

public:


    bool loadData(py::array_t<double, py::array::forcecast> arr, py::array_t<double, py::array::forcecast> lbls)
    {
        std::cout<<"[Loading Data "
                   "]"<<std::endl;
        ASSERT_THROW( (arr.ndim() == 2), "Expected two-dimensional Data array");
        ASSERT_THROW( (lbls.ndim() == 1), "Expected one-dimensional Label array");
        auto arr_buf = arr.request();
        auto lbl_buf = lbls.request();
        data = std::unique_ptr<DataPointCollection>(new DataPointCollection());

        std::vector<float> row;
        data->reserve(arr.shape(0),arr.shape(1));
        //test_train_data->dimension_ = arr.shape(1);

        int count = 0;

        for(int i = 0;i<arr.shape(0);i++)
        {
            for(int j=0;j<arr.shape(1);j++)
                data->putValue((float)arr.at(i,j), (int)lbls.at(i), i,j);
//                std::cout<<arr.at(i,j);
            count++;
        }

        std::cout<<"[Loading data DONE]"<<std::endl;

        return (count == data->Count());
    }

    bool modelExists()
    {
        return (forest.get() != NULL);
    }

    // Test function
    int add(int i, int j) {
        return i + j;
    }

    bool loadModel(std::string filename)
    {
        forest = Forest<LinearFeatureResponseSVM, HistogramAggregator>::DeserializeJson(filename);

        return true;
    }

    bool saveModel(std::string filename)
    {
        forest->SerializeJson(filename);

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
        return true;
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


    bool setFeatureMask(int i)
    {
        trainingParameters.featureMask = static_cast<FeatureMaskType>(i);
        return true;
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


    py::array_t<double> onlyTest()
    {
        std::vector<HistogramAggregator> distbns;
        ClassificationDemo<LinearFeatureResponseSVM>::Test(*forest.get(),
                                                           *data.get(),
                                                           distbns);

        int nr_classes = data->CountClasses();


        int dim_1 = (int) distbns.size();
        int dim_2 = nr_classes;
//        np::ndarray result = np::zeros(bp::make_tuple(distbns.size(),nr_classes), np::dtype::get_builtin<double>());
        py::array_t<double, py::array::c_style> result_data({dim_1, dim_2});
        auto r = result_data.mutable_unchecked<2>();

        for (int i=0; i<distbns.size(); i++) {
            for (int j = 0; j < nr_classes; j++)
                r(i, j) = distbns[i].GetProbability(j);
        }

        return result_data;
    }

};



PYBIND11_MODULE(slither_py, m) {
    m.doc() = "slither -  a module to do awesome things";
//    m.def("add", &add, "Simple function to add");
    py::class_<SlitherWrapper> slitherWrapObj(m, "slither");
    slitherWrapObj.def(py::init());
    slitherWrapObj.def("add", &SlitherWrapper::add);
    slitherWrapObj.def("loadModel", &SlitherWrapper::loadModel);
    slitherWrapObj.def("saveModel", &SlitherWrapper::saveModel);
    slitherWrapObj.def("setDefaultParams", &SlitherWrapper::setDefaultParams);
    slitherWrapObj.def("setParams", &SlitherWrapper::setParams);
    slitherWrapObj.def("onlyTrain", &SlitherWrapper::onlyTrain);
    slitherWrapObj.def("onlyTest", &SlitherWrapper::onlyTest);
    slitherWrapObj.def("loadData", &SlitherWrapper::loadData);
    slitherWrapObj.def("modelExists", &SlitherWrapper::modelExists);
    slitherWrapObj.def("setThreads", &SlitherWrapper::setThreads);
    slitherWrapObj.def("setFeatureMask", &SlitherWrapper::setFeatureMask);
    slitherWrapObj.def("setMaxDecisionLevels", &SlitherWrapper::setMaxDecisionLevels);
    slitherWrapObj.def("setNumberOfCandidateFeatures",&SlitherWrapper::setNumberOfCandidateFeatures);
    slitherWrapObj.def("setNumberOfThresholds",&SlitherWrapper::setNumberOfThresholds);
    slitherWrapObj.def("setTrees",&SlitherWrapper::setTrees);
    slitherWrapObj.def("setQuiet",&SlitherWrapper::setQuiet);
    slitherWrapObj.def("setSVM_C",&SlitherWrapper::setSVM_C);
}
