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
#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>



using namespace Slither;

void DisplayTextFiles(const std::string& relativePath);


int discoverDims(std::string filename);

std::unique_ptr<DataPointCollection> LoadTrainingData(const std::string& filename, const std::string& model_name,
                                                     Eigen::VectorXf& biases, Eigen::VectorXf& divisors);
std::unique_ptr<DataPointCollection> LoadTestingData(const std::string& filename, const std::string& model_name,
                                                    Eigen::VectorXf& biases, Eigen::VectorXf& divisors);




int data_dimensions = 3;
TrainingParameters trainingParameters;
std::string dummy = "";
std::string hardcoded_location = "/home/prassanna/Projects/Slither/";
std::string train_filename = hardcoded_location + "data/sclf/sample_train.txt";
std::string test_filename = hardcoded_location + "data/sclf/sample_train.txt";
std::string predict_filename = hardcoded_location + "data/sclf/sample_predict.txt";
//float svm_c = 0.5;
std::string mode = "Standard";
bool train_flag = false;
bool test_flag = false;
std::string forest_loc ="forest_400.out";
bool scale_flag = false;
const std::vector<std::string> FeatureNames = {"Standard", "Hypercolumn", "LBP", "Fisher", "Hypercolumn+Location", "2Layer Hypercolumn", "Hypercolumn+Location+Color"};

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



  CLI::App app{"Slither Random Forest - A Random Forest library with SVM local experts"};
  
  // File options
  app.add_option("--train", train_filename, "Training Data file (CSV TAB DELIMITED)");
  app.add_option("--test", test_filename, "Testing Data file");
  app.add_option("--predict", predict_filename, "Predicted output file - Will be (over)written");
  app.add_option("--model", forest_loc, "Where to dump or load the trained forest");
  
  // Data parameters
  app.add_option("--dims", data_dimensions, "Dimensionality of data (Nr. of attributes)");
  
  // Forest parameters
  int num_trees = 10;
  int max_depth = 15;
  int num_features = 10;
  int num_thresholds = 10;
  float svm_c = 0.5f;
  app.add_option("--trees", num_trees, "Number of Trees in the forest");
  app.add_option("--depth", max_depth, "Number of Decision Levels");
  app.add_option("--feats", num_features, "Number of times to randomly choose a candidate feature");
  app.add_option("--thresh", num_thresholds, "Number of times to sample the threshold");
  app.add_option("--svm_c", svm_c, "C Parameter of the SVM");
  
  // Operation parameters
  bool verbose = true;
  std::string mode = "Standard";
  std::string op_mode = "tr-te";
  int mask_type = 1;
  int num_threads = 1;
  bool scale_data = false;
  bool use_parallel = false;
  
  app.add_option("--verbose", verbose, "Display output");
  app.add_option("--mode", mode, "Random Forest operating mode");
  app.add_option("--op_mode", op_mode, "train | test | tr-te");
  app.add_option("--mask_type", mask_type, "standard=0, hypercolumn=1, lbp=2, fisher=3, hypercolumn_loc=4, hypercolumn2=5, hypercolumn_loc_color=6, hypercolumn_lbp_loc_color=7");
  app.add_option("--threads", num_threads, "Max. Threads for training the forest");
  app.add_option("--scale", scale_data, "Should I scale the data");
  app.add_option("--parallel", use_parallel, "Should I use parallel training");

  CLI11_PARSE(app, argc, argv);
  
  // Set training parameters from CLI11 parsed values
  trainingParameters.NumberOfTrees = num_trees;
  trainingParameters.MaxDecisionLevels = max_depth;
  trainingParameters.NumberOfCandidateFeatures = num_features;
  trainingParameters.NumberOfCandidateThresholdsPerFeature = num_thresholds;
  trainingParameters.svm_c = svm_c;
  trainingParameters.Verbose = verbose;
  trainingParameters.featureMask = (FeatureMaskType)mask_type;
  trainingParameters.maxThreads = num_threads;
  scale_flag = scale_data;
  
  // Set train/test flags based on op_mode
  if(op_mode == "train") {
    train_flag = true;
    test_flag = false;
  } else if(op_mode == "test") {
    train_flag = false;
    test_flag = true;
  } else if(op_mode == "tr-te") {
    train_flag = true;
    test_flag = true;
  } else {
    std::cout << "Unknown operation mode: " << op_mode << ". Using train-test mode." << std::endl;
    train_flag = true;
    test_flag = true;
  }
  
  printParsedArguments();








/*
  if (trainingData.get()==0)
       return 0; // LoadTrainingData() generates its own progress/error messages
    */
  Eigen::VectorXf divisors, biases;
  std::unique_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > forest;
  if(train_flag)
  {
    data_dimensions = discoverDims (train_filename);
    std::unique_ptr<DataPointCollection> trainingData
            = std::unique_ptr<DataPointCollection> ( LoadTrainingData(train_filename, forest_loc, biases, divisors) );

    LinearFeatureSVMFactory linearFeatureFactory;
    if(!use_parallel)
      if(num_threads>1)
        forest
              = ClassificationDemo<LinearFeatureResponseSVM>::Train(*trainingData,
                                                                    &linearFeatureFactory,
                                                                    trainingParameters);
      else
        forest = ClassificationDemo<LinearFeatureResponseSVM>::TrainSingle(*trainingData,
                                                                     &linearFeatureFactory,
                                                                     trainingParameters);
    else
      forest = ClassificationDemo<LinearFeatureResponseSVM>::TrainParallel(*trainingData,
                                                                         &linearFeatureFactory,
                                                                         trainingParameters);

    //Testing out regression
    //std::unique_ptr<Forest<LinearFeatureResponseSVM, LinearFitAggregator1d> > forest2 = RegressionExample::Train(
    //      *trainingData.get(), trainingParameters);

    //forest->SerializeJson(forest_loc);
  }


  if(test_flag)
  {
    //data_dimensions = discoverDims (test_filename);
    std::unique_ptr<DataPointCollection> testdata
            = std::unique_ptr<DataPointCollection> ( LoadTestingData(test_filename,forest_loc,  biases, divisors) );

    //std::unique_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > trained_forest_loaded =
    //    Forest<LinearFeatureResponseSVM, HistogramAggregator>::DeserializeJson(forest_loc);


    // std::vector<HistogramAggregator> distbns;
    // ClassificationDemo<LinearFeatureResponseSVM>::Test(*trained_forest_loaded.get(),
    //                                                    *testdata.get(),
    //                                                    distbns);
    // 
    // std::cout<<"[WRITING PREDICTED DATA]"<<std::endl;
    // //writePredData (predict_filename, distbns);
    // trained_forest_loaded.release();
    // distbns.clear();
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




void printParsedArguments()
{
  std::cout << "[PARSED ARGUMENTS]" << std::endl;
  std::cout << "Training Data: " << train_filename << std::endl;
  std::cout << "Testing Data: " << test_filename << std::endl;
  std::cout << "Predicted output: " << predict_filename << std::endl;
  std::cout << "Forest Location: " << forest_loc << std::endl;
  std::cout << "Data Dimensions: " << data_dimensions << std::endl;
  std::cout << "Number of Trees: " << trainingParameters.NumberOfTrees << std::endl;
  std::cout << "Decision Levels: " << trainingParameters.MaxDecisionLevels << std::endl;
  std::cout << "Candidate Features: " << trainingParameters.NumberOfCandidateFeatures << std::endl;
  std::cout << "Candidate Thresholds: " << trainingParameters.NumberOfCandidateThresholdsPerFeature << std::endl;
  std::cout << "SVM C Parameter: " << trainingParameters.svm_c << std::endl;
  std::cout << "Verbose: " << trainingParameters.Verbose << std::endl;
  std::cout << "[FINISHED PARSING]" << std::endl << std::endl;
}

/*
    std::cout << "\t Training data source was set to ";
  else
    std::cout << "\t Training Data source was not set. Using Default...";
  train_filename = vm["train"].as<std::string>();
  std::cout<<"<"<<train_filename<<">"<<std::endl;


  std::cout<<"2. [Testing Data]";
  if (vm.count("test"))
    std::cout << "\t Testing data source was set to ";
  else
    std::cout << "\t Testing Data source was not set. Using Default...";
  test_filename = vm["test"].as<std::string>();
  std::cout<<"<"<<test_filename<<">"<<std::endl;

  std::cout << "3. [Predicted output]";
  if (vm.count ("predict"))
    std::cout << "\t Predicted output filename was set to ";
  else
    std::cout << "\t Predicted output filename  was not set. Using Default...";
  predict_filename = vm["predict"].as<std::string> ();
  std::cout << "<" << predict_filename << ">" << std::endl;

  std::cout<<"4. [Forest Location]";
  if (vm.count("model"))
    std::cout << "\t Forest Location source was set to ";
  else
    std::cout << "\t Forest Location source was not set. Using Default...";
  forest_loc = vm["model"].as<std::string>();
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
  if (vm.count("depth"))
    std::cout << "\t Number of Decision levels is set to ";
  else
    std::cout << "\t Number of Decision levels not set. Using Default...";
  trainingParameters.MaxDecisionLevels = vm["depth"].as<int>();
  std::cout<<"<"<<trainingParameters.MaxDecisionLevels<<">"<<std::endl;

  std::cout<<"8. [Candidate Features]";
  if (vm.count("feats"))
    std::cout << "\t Number of Canidate Features is set to ";
  else
    std::cout << "\t Number of Canidate Features was not set. Using Default...";
  trainingParameters.NumberOfCandidateFeatures = vm["feats"].as<int>();
  std::cout<<"<"<<trainingParameters.NumberOfCandidateFeatures<<">"<<std::endl;

  std::cout<<"9. [Candidate Thresholds]";
  if (vm.count("thresh"))
    std::cout << "\t Number of Canidate Thresholds is set to ";
  else
    std::cout << "\t Number of Canidate Thresholds was not set. Using Default...";
  trainingParameters.NumberOfCandidateThresholdsPerFeature = vm["thresh"].as<int>();
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
    std::cout << "\t Mask type was not set. Using Default...";
  int mask_type = vm["mask_type"].as<int>();
  std::cout<<mask_type <<"-->"<<FeatureNames[mask_type]<<std::endl;
  trainingParameters.featureMask = static_cast<FeatureMaskType >(mask_type);


  std::cout<<"15. [Max Threads ]";
  if (vm.count("threads"))
    std::cout << "\t Max Threads was set to : ";
  else
    std::cout << "\t Max Threads was not set. Using Default...";
  int maxThreads = vm["threads"].as<int>();
  std::cout<<maxThreads<<std::endl;
  trainingParameters.maxThreads = maxThreads;


  std::cout<<"16. [Scale Data ]";
  if (vm.count("threads"))
    std::cout << "\t Scale Data Flag was set to : ";
  else
    std::cout << "\t Scale Data Flag was not set. Using Default...";
  scale_flag = vm["scale"].as<bool>();
  std::cout<<scale_flag<<std::endl;


  std::cout<<"[FINISHED PARSING]"<<std::endl<<std::endl;


}
*/
std::unique_ptr<DataPointCollection> LoadTrainingData(
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
      return std::unique_ptr<DataPointCollection>(0);
    }

    path = path + alternativePath;

    r.open(path.c_str());

    if(r.fail())
    {
      std::cout << "Failed to open either \"" << filename << "\" or \"" << path.c_str() << "\"." << std::endl;
      return std::unique_ptr<DataPointCollection>(0);
    }
  }

  std::unique_ptr<DataPointCollection> trainingData;
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
    return std::unique_ptr<DataPointCollection>(0);
  }

  if (trainingData->Count() < 1)
  {
    std::cout << "Insufficient training data." << std::endl;
    return std::unique_ptr<DataPointCollection>(0);
  }

  return trainingData;
}*/


std::unique_ptr<DataPointCollection> LoadTrainingData(const std::string& filename, const std::string& model_name,
                                                     Eigen::VectorXf& biases, Eigen::VectorXf& divisors)
{
  std::string path;




  std::unique_ptr<DataPointCollection> trainingData;
  trainingData = trainingData->Load(filename);

  if(scale_flag)
  {
    std::vector<float> bvec;
    std::vector<float> dvec;
    trainingData->scaleData(bvec, dvec);
    biases = Eigen::Map<Eigen::VectorXf>(bvec.data(), bvec.size());
    divisors = Eigen::Map<Eigen::VectorXf>(dvec.data(), dvec.size());

    nlohmann::json j;
    j["Bias"] = bvec;
    j["Weights"] = dvec;
    std::ofstream ofs(model_name + "_dataWeights.json");
    ofs << j.dump(2);
  }
  return trainingData;
}


std::unique_ptr<DataPointCollection> LoadTestingData(const std::string& filename, const std::string& model_name,
                                                    Eigen::VectorXf& biases, Eigen::VectorXf& divisors)
{
  std::unique_ptr<DataPointCollection> trainingData;
  trainingData = trainingData->Load(filename);

  if(scale_flag)
  {
    std::ifstream ifs(model_name + "_dataWeights.json");
    if(ifs.good())
    {
      nlohmann::json j;
      ifs >> j;
      std::vector<float> bvec = j["Bias"].get<std::vector<float>>();
      std::vector<float> dvec = j["Weights"].get<std::vector<float>>();
      biases = Eigen::Map<Eigen::VectorXf>(bvec.data(), bvec.size());
      divisors = Eigen::Map<Eigen::VectorXf>(dvec.data(), dvec.size());
      trainingData->doScaleData(biases, divisors);
    }
  }




  //trainingData->showMat();
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

