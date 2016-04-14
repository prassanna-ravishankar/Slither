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

using namespace MicrosoftResearch::Cambridge::Sherwood;
namespace po = boost::program_options;

void parseArguments(po::variables_map& vm);

void DisplayTextFiles(const std::string& relativePath);


int discoverDims(std::string filename);

std::auto_ptr<DataPointCollection> LoadTrainingData(
        const std::string& filename,
        const std::string& alternativePath,
        int dimension,
        DataDescriptor::e descriptor);



//HARDCODING DEFAULTS - get from boost argparse
int data_dimensions = 3;
TrainingParameters trainingParameters;
std::string dummy = "";
std::string train_filename = "../demo/data/sclf/exp1_n2.txt";
std::string test_filename = "../demo/data/sclf/exp1_n2.txt";
std::string predict_filename = "../demo/data/sclf/sample_predict.txt";
//float svm_c = 0.5;
std::string mode = "Standard";
bool train_flag = false;
bool test_flag = false;
std::string forest_loc ="forest.out";


int main(int argc, char* argv[])
{

  //Defaults
  trainingParameters.MaxDecisionLevels = 10;
  trainingParameters.NumberOfCandidateFeatures = 10;
  trainingParameters.NumberOfCandidateThresholdsPerFeature = 10;
  trainingParameters.NumberOfTrees = 10;
  trainingParameters.Verbose = true;
  trainingParameters.igType =  ig_shannon;
  trainingParameters.featureMask = FeatureMaskType::hypercolumn;



  po::options_description desc("Allowed Options");
  desc.add_options()
          ("help,h", "produce help message")
          ("data_train",po::value<std::string>()->default_value(train_filename), "Training Data file (CSV TAB DELIMITED)")
          ("data_test",po::value<std::string>()->default_value(test_filename), "Testing Data file")
          ("data_predict",po::value<std::string>()->default_value(predict_filename), "Predicted output file - Will be (over)written")
          ("forest_loc",po::value<std::string>()->default_value(forest_loc), "Where to dump  or load the trained forest")
          ("dims",po::value<int>()->default_value(data_dimensions), "Dimensionality of data (Nr. of attributes)")
          ("trees",po::value<int>()->default_value(50), "Number of Trees in the forest")
          ("dlevels",po::value<int>()->default_value(10), "Number of Decision Levels")
          ("candidate_feats",po::value<int>()->default_value(50), "Number of times to randomly choose a candidate feature")
          ("candidate_thresh",po::value<int>()->default_value(50), "Number of times to sample the threshold")
          ("svm_c",po::value<float>()->default_value(0.5), "C Parameter of the SVM")
          ("verbose",po::value<bool>()->default_value(true), "Display output")
          ("mode",po::value<std::string>()->default_value("Standard"), "Random Forest operating mode")
          ("op_mode",po::value<std::string>()->default_value("tr-te"), "train | test | tr-te")
          ("mask_type",po::value<int>()->default_value(0), "standard=0, hypercolumn=1, lbp=2, fisher=3")
          ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if ( (vm.count("help"))) {
    std::cout << desc << "\n";
    return 1;
  }

  parseArguments(vm);








/*
  if (trainingData.get()==0)
       return 0; // LoadTrainingData() generates its own progress/error messages
    */

  if(train_flag)
  {
    data_dimensions = discoverDims (train_filename);
    std::auto_ptr<DataPointCollection> trainingData
            = std::auto_ptr<DataPointCollection> ( LoadTrainingData(train_filename,
                                                                    dummy,
                                                                    data_dimensions,
                                                                    DataDescriptor::HasClassLabels ) );

    LinearFeatureSVMFactory linearFeatureFactory;
    std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > forest
            = ClassificationDemo<LinearFeatureResponseSVM>::Train(*trainingData,
                                                                  &linearFeatureFactory,
                                                                  trainingParameters);

    //Testing out regression
    //std::auto_ptr<Forest<LinearFeatureResponseSVM, LinearFitAggregator1d> > forest2 = RegressionExample::Train(
    //      *trainingData.get(), trainingParameters);

    forest->Serialize(forest_loc);
    forest.release();
  }


  if(test_flag)
  {
    data_dimensions = discoverDims (test_filename);
    std::auto_ptr<DataPointCollection> testdata
            = std::auto_ptr<DataPointCollection> ( LoadTrainingData(test_filename,
                                                                    dummy,
                                                                    data_dimensions,
                                                                    DataDescriptor::HasClassLabels ) );

    std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > trained_forest
            = Forest<LinearFeatureResponseSVM, HistogramAggregator>::Deserialize(forest_loc);


    std::vector<HistogramAggregator> distbns;
    ClassificationDemo<LinearFeatureResponseSVM>::Test(*trained_forest.get(),
                                                       *testdata.get(),
                                                       distbns);

    std::cout<<"[WRITING PREDICTED DATA]"<<std::endl;
    //writePredData (predict_filename, distbns);
    trained_forest.release();
    distbns.clear();
  }


  return 0;

}



int discoverDims(std::string filename)
{
  std::ifstream FILE(filename);
  std::string line;
  if(!FILE.is_open ())
    return -1;


  //FILE>>line;
  getline (FILE,line);
  std::vector<char> data(line.begin(), line.end());
  int count =  std::count(data.begin (),data.end (), '\t');
  getline (FILE,line);
  data = std::vector<char> (line.begin(), line.end());
  int count2 =  std::count(data.begin (),data.end (), '\t');
  FILE.close ();

  std::cout<<"Discovered Dimensions are : "<<count * (count==count2)<<std::endl;
  return count * (count==count2);
}




void parseArguments(po::variables_map& vm)
{

  std::cout<<"[PARSING ARGUMENTS] "<<std::endl;

  //Data_Train
  std::cout<<"2. [Training Data]";
  if (vm.count("Training Data"))
    std::cout << "\t Training data source was set to ";
  else
    std::cout << "\t Training Data source was not set. Using Default...";
  train_filename = vm["data_train"].as<std::string>();
  std::cout<<"<"<<train_filename<<">"<<std::endl;


  std::cout<<"2. [Testing Data]";
  if (vm.count("data_test"))
    std::cout << "\t Testing data source was set to ";
  else
    std::cout << "\t Testing Data source was not set. Using Default...";
  test_filename = vm["data_test"].as<std::string>();
  std::cout<<"<"<<test_filename<<">"<<std::endl;

  std::cout << "3. [Predicted output]";
  if (vm.count ("data_predict"))
    std::cout << "\t Predicted output filename was set to ";
  else
    std::cout << "\t Predicted output filename  was not set. Using Default...";
  predict_filename = vm["data_predict"].as<std::string> ();
  std::cout << "<" << predict_filename << ">" << std::endl;

  std::cout<<"4. [Forest Location]";
  if (vm.count("forest_loc"))
    std::cout << "\t Forest Location source was set to ";
  else
    std::cout << "\t Forest Location source was not set. Using Default...";
  forest_loc = vm["forest_loc"].as<std::string>();
  std::cout<<"<"<<forest_loc<<">"<<std::endl;


  std::cout<<"5. [Dimensionality of the data]";
  if (vm.count("dims"))
    std::cout << "\t Number of Dimensions of data is set to ";
  else
    std::cout << "\t Number of Dimensions of data was not set. Using Default...";
  data_dimensions = vm["dims"].as<int>();
  std::cout<<"<"<<data_dimensions<<">"<<std::endl;


  std::cout<<"6. [Number of Trees]";
  if (vm.count("trees"))
    std::cout << "\t Number of Trees is set to ";
  else
    std::cout << "\t Number of Trees was not set. Using Default...";
  trainingParameters.NumberOfTrees = vm["trees"].as<int>();
  std::cout<<"<"<<trainingParameters.NumberOfTrees<<">"<<std::endl;

  std::cout<<"7. [Decision Levels]";
  if (vm.count("dlevels"))
    std::cout << "\t Number of Decision levels is set to ";
  else
    std::cout << "\t Number of Decision levels not set. Using Default...";
  trainingParameters.MaxDecisionLevels = vm["dlevels"].as<int>();
  std::cout<<"<"<<trainingParameters.MaxDecisionLevels<<">"<<std::endl;

  std::cout<<"8. [Candidate Features]";
  if (vm.count("candidate_feats"))
    std::cout << "\t Number of Canidate Features is set to ";
  else
    std::cout << "\t Number of Canidate Features was not set. Using Default...";
  trainingParameters.NumberOfCandidateFeatures = vm["candidate_feats"].as<int>();
  std::cout<<"<"<<trainingParameters.NumberOfCandidateFeatures<<">"<<std::endl;

  std::cout<<"9. [Candidate Thresholds]";
  if (vm.count("candidate_thresh"))
    std::cout << "\t Number of Canidate Thresholds is set to ";
  else
    std::cout << "\t Number of Canidate Thresholds was not set. Using Default...";
  trainingParameters.NumberOfCandidateThresholdsPerFeature = vm["candidate_thresh"].as<int>();
  std::cout<<"<"<<trainingParameters.NumberOfCandidateThresholdsPerFeature<<">"<<std::endl;

  std::cout<<"10. [SVM_C]";
  if (vm.count("svm_c"))
    std::cout << "\t C Param of SVM is set to ";
  else
    std::cout << "\t C Param of SVM was not set. Using Default...";
  trainingParameters.svm_c = vm["svm_c"].as<float>();
  std::cout<<"<"<<trainingParameters.svm_c<<">"<<std::endl;


  std::cout<<"11. [Verbosity]";
  if (vm.count("verbose"))
    std::cout << "\t Verbosity is set to ";
  else
    std::cout << "\t Verbosity was not set. Using Default...";
  trainingParameters.Verbose = vm["verbose"].as<bool>();
  std::cout<<"<"<<trainingParameters.Verbose<<">"<<std::endl;


  std::cout<<"12. [Computing Mode ]";
  if (vm.count("mode"))
    std::cout << "\t Mode is set to ";
  else
    std::cout << "\t Mode was not set. Using Default...";
  mode = vm["mode"].as<std::string>();
  std::cout<<"<"<<mode<<">"<<std::endl;

  std::cout<<"13. [Operating Mode ]";
  if (vm.count("op_mode"))
    std::cout << "\t Operating  Mode is set to ";
  else
    std::cout << "\t Operating  Mode was not set. Using Default...";
  std::string op_mode = vm["op_mode"].as<std::string>();

  if(op_mode.compare("train")==0) {
    train_flag = true;
    std::cout<<"<TRAIN>"<<std::endl;
  }

  else if(op_mode.compare("test")==0) {
    test_flag = true;
    std::cout << "<TEST>" << std::endl;
  }
  else if(op_mode.compare("tr-te")==0)
  {
    train_flag=true;
    test_flag = true;
    std::cout<<"<TRAIN-TEST>"<<std::endl;
  }
  else
    std::cout<<"Couldn't parse train-test mode. Doing both"<<std::endl;

  std::cout<<"14. [Mask Type ]";
  if (vm.count("mask_type"))
    std::cout << "\t Mask type was set to ";
  else
    std::cout << "\t Operating  Mode was not set. Using Default...";
  int mask_type = vm["mask_type"].as<int>();
  trainingParameters.featureMask = static_cast<FeatureMaskType >(mask_type);

  std::cout<<"[FINISHED PARSING]"<<std::endl<<std::endl;


}


std::auto_ptr<DataPointCollection> LoadTrainingData(
        const std::string& filename,
        const std::string& alternativePath,
        int dimension,
        DataDescriptor::e descriptor)
{
  std::ifstream r;

  r.open(filename.c_str());
  int dims = 0;
  std::string line;
  r>>line;
  r.close ();

  r.open(filename.c_str());

  if(r.fail())
  {
    std::string path;

    try
    {
      path = GetExecutablePath();
    }
    catch(std::runtime_error& e)
    {
      std::cout<< "Failed to determine executable path. " << e.what();
      return std::auto_ptr<DataPointCollection>(0);
    }

    path = path + alternativePath;

    r.open(path.c_str());

    if(r.fail())
    {
      std::cout << "Failed to open either \"" << filename << "\" or \"" << path.c_str() << "\"." << std::endl;
      return std::auto_ptr<DataPointCollection>(0);
    }
  }

  std::auto_ptr<DataPointCollection> trainingData;
  try
  {
    trainingData = DataPointCollection::Load (
            r,
            dimension,
            descriptor );
  }
  catch (std::runtime_error& e)
  {
    std::cout << "Failed to read training data. " << e.what() << std::endl;
    return std::auto_ptr<DataPointCollection>(0);
  }

  if (trainingData->Count() < 1)
  {
    std::cout << "Insufficient training data." << std::endl;
    return std::auto_ptr<DataPointCollection>(0);
  }

  return trainingData;
}

void DisplayTextFiles(const std::string& relativePath)
{
  std::string path;

  try
  {
    path = GetExecutablePath();
  }
  catch(std::runtime_error& e)
  {
    std::cout<< "Failed to find demo data files. " << e.what();
    return;
  }

  path = path + relativePath;

  std::vector<std::string> filenames;

  try
  {
    GetDirectoryListing(path, filenames, ".txt");
  }
  catch(std::runtime_error& e)
  {
    std::cout<< "Failed to list demo data files. " << e.what();
    return;
  }

  if (filenames.size() > 0)
  {
    std::cout << "The following demo data files can be specified as if they were on your current path:-" << std::endl;

    for(std::vector<std::string>::size_type i=0; i<filenames.size(); i++)
      std::cout << "  " << filenames[i].c_str() << std::endl;
  }
}

