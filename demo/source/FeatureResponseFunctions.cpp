#include "FeatureResponseFunctions.h"

#include <cmath>

#include <sstream>

#include "DataPointCollection.h"
#include "Random.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  AxisAlignedFeatureResponse AxisAlignedFeatureResponse ::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
  {
    return AxisAlignedFeatureResponse(random.Next(0, 2));
  }

  float AxisAlignedFeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const
  {
    const DataPointCollection& concreteData = (DataPointCollection&)(data);
    return concreteData.GetDataPoint((int) sampleIndex).at<float>(axis_);
  }

  std::string AxisAlignedFeatureResponse::ToString() const
  {
    std::stringstream s;
    s << "AxisAlignedFeatureResponse(" << axis_ << ")";

    return s.str();
  }

  /// <returns>A new LinearFeatureResponse2d instance.</returns>
  LinearFeatureResponse2d LinearFeatureResponse2d::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
  {
    double dx = 2.0 * random.NextDouble() - 1.0;
    double dy = 2.0 * random.NextDouble() - 1.0;

    double magnitude = sqrt(dx * dx + dy * dy);

    return LinearFeatureResponse2d((float)(dx / magnitude), (float)(dy / magnitude));
  }

  float LinearFeatureResponse2d::GetResponse(const IDataPointCollection& data, unsigned int index) const
  {
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    cv::Mat rowMat = concreteData.GetDataPoint((int) index);
    //std::cout<<rowMat<<" ";
    //float bla = dx_ * rowMat.at<float>(0) + dy_ * rowMat.at<float>(1);
    //std::cout<<bla<<std::endl;
    //return dx_ * concreteData.GetDataPoint((int)index)[0] + dy_ * concreteData.GetDataPoint((int)index)[1];
    return dx_ * rowMat.at<float>(0) + dy_ * rowMat.at<float>(1);
  }

  std::string LinearFeatureResponse2d::ToString() const
  {
    std::stringstream s;
    s << "LinearFeatureResponse(" << dx_ << "," << dy_ << ")";

    return s.str();
  }


  /// <returns>A new LinearFeatureResponse instance.</returns>
  LinearFeatureResponse LinearFeatureResponse::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
  {
    LinearFeatureResponse lr;
    //lr.dimensions_ = data.GetDimension();
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    lr.dimensions_ = concreteData.Dimensions();
    lr.vWeights_.resize(lr.dimensions_,-1);

    double magnitude = 0.0f;
    for (int i=0; i<lr.dimensions_; i++)
    {
      double rnd = 2.0 * random.NextDouble() - 1.0;
      magnitude += rnd*rnd;
      lr.vWeights_[i] = (float)rnd;
    }
    magnitude = sqrt(magnitude);

    for (int i=0; i<lr.dimensions_; i++)
      lr.vWeights_[i] /= (float)magnitude;

    lr.dimensions_ = concreteData.Dimensions();
    return lr;
  }

  float LinearFeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int index) const
  {
    // Multiply the weights by the vector to classify and sum
    //return vec4::Dot(&vWeights_[0], ((const DataPointCollection&)(data)).GetDataPoint((int)index), dimensions_) + bias_;
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    //std::vector<float> rowData = concreteData.GetDataPointRange(index);
    //float response = std::inner_product(rowData.begin(),rowData.end(), vWeights_.begin(), bias_);
      float response = 0.0;
       return response;

  }

  std::string LinearFeatureResponse::ToString() const
  {
    std::stringstream s;
    s << "LinearFeatureResponse(";
    s << vWeights_[0];
    for (int i=1; i<dimensions_; i++)
      s << "," << vWeights_[i];
    s << ")";

    return s.str();
  }




//MAIN STUFF
  void LinearFeatureResponseSVM::GenerateMask(Random &random, std::vector<int>& vIndex, int dims, bool root_node)
  {

    int numBloks = random.Next(dims/2, dims+1);

    for(int i=0;i<numBloks;i++)
    {
      int indx = random.Next(0,dims);
      vIndex.push_back(indx);
    }

  }

  void LinearFeatureResponseSVM::GenerateMaskLBP(Random &random, std::vector<int>& vIndex, int dims, bool root_node)
  {
    int maxBloks = 12;
    int feature_length = 59;
    int minBloks = 10;

    int numBloks = random.Next(1,maxBloks+1);


    for (int i = 0; i< numBloks;i ++ )
    {
      int idx = random.Next(0,maxBloks);
      int feature_start = idx * feature_length;
      for(int j = feature_start; j<feature_start+feature_length;j++)
      {
        vIndex.push_back(j);
      }
    }

  }

  void LinearFeatureResponseSVM::GenerateMaskFisher(Random &random, std::vector<int>& vIndex, int dims, bool root_node)
  {

    bool machine_choice = (random.NextDouble()>0.5);
    bool loc_choice = random.NextDouble()>0.5;
    int numBloks = 0;
    int maxBloks = 1;

    if(machine_choice)
    {
      numBloks = random.Next(1,HYPER_MACHINE_PAIRS);
      maxBloks = HYPER_MACHINE_PAIRS;
    }
    else
    {
      numBloks = random.Next(1, HYPER_LBP_PAIRS);
      maxBloks = HYPER_LBP_PAIRS;
    }

    bool lbp_choice = !machine_choice;

    for(int i=0;i<numBloks;i++)
    {
      int indx = random.Next(0,maxBloks);

      //Selecting mean
      vIndex.push_back( (HYPERFISHER_MACHINE_DIM*lbp_choice) + indx );

      //Selecting std_dev
      vIndex.push_back( (HYPERFISHER_MACHINE_DIM*lbp_choice) + (indx + (HYPER_MACHINE_PAIRS*machine_choice + HYPER_LBP_PAIRS*lbp_choice)) );
    }

    if(loc_choice)
    {
      vIndex.push_back(HYPERFISHER_MACHINE_DIM+HYPER_LBP_DIM);
      vIndex.push_back(HYPERFISHER_MACHINE_DIM+HYPER_LBP_DIM+1);
    }

  }


  void LinearFeatureResponseSVM::GenerateMaskHypercolumn(Random &random, std::vector<int> &vIndex, int dims,
                                                         bool root_node)
  {

    //Discarding LBP and position to check
    int numBloks = random.Next (5, 15);
    for(int i=0;i<numBloks;i++)
    {
      int idx = random.Next (0,NN_DIM);
      vIndex.push_back (idx);
      vIndex.push_back (idx+NN_DIM);
    }

  }


  LinearFeatureResponseSVM LinearFeatureResponseSVM::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
  {
    LinearFeatureResponseSVM lr;
    DataPointCollection& concreteData = (DataPointCollection&)(data);
    //this->dimensions_ =  concreteData.Dimensions();
    lr.dimensions_ = concreteData.Dimensions();


    switch(featureMask)
    {
      case fisher:GenerateMaskFisher (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
        break;
      case lbp:GenerateMaskLBP (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
        break;
      case hypercolumn: GenerateMaskHypercolumn (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
        break;
      case standard:GenerateMask (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
        break;
      default: std::cout<<"Using unknown mask function. Re-check parameters"<<std::endl;
    }

    std::sort(lr.vIndex_.begin(), lr.vIndex_.end());

    cv::Ptr<cvml::SVM> svm;
    svm = cvml::SVM::create();
    svm->setType(cvml::SVM::C_SVC);
    svm->setKernel(cvml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 1000, 0.01));


    svm->setC(svm_c);

    cv::Ptr<cvml::TrainData> tdata = concreteData.getTrainDataWithMask(lr.vIndex_,i0,i1);


    svm->train(tdata);

    lr.nWeights_ = lr.vIndex_.size();
    lr.vWeights_.resize(lr.vIndex_.size(),1); //Initializing weights as unit vector
    lr.bias_ = 0; // 0 ->bias


    if(svm->isTrained())
    {
      cv::Mat alpha, wts;

      lr.bias_ = -1 * (float)svm->getDecisionFunction(0, alpha, wts);
      cv::Mat svs = svm->getSupportVectors();
      for(int j=0;j<svs.cols;j++)
        lr.vWeights_[j]=(svs.at<float>(j));
    }
    svm.release();

    //lr.vWeights_.resize()

    //cv::Mat svs = lr.svm->getSupportVectors();
    //svm->getDecisionFunction(0,alpha,svs);
    //std::cout<<"[DEBUG : "<<i0<<" -->"<<i1<<"]    "<<tdata->getNSamples()<<std::endl;
    //std::cout<<"[DEBUG : FeatureResponseSVM / isTrained?]"<<lr.nWeights_<<" "<<lr.svm->isTrained()<<std::endl;







    return lr;
  }

  float LinearFeatureResponseSVM::GetResponse(const IDataPointCollection& data, unsigned int index) const
  {
    //std::cout<<"At feature response 1"<<std::endl;

    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    cv::Mat rowMat = concreteData.GetDataPoint(index);
    std::vector<float> rowVector;

    for (int i = 0;i<nWeights_;i++)
       rowVector.push_back(rowMat.at<float>(vIndex_[i]));

    double response = std::inner_product(rowVector.begin(), rowVector.end(), vWeights_.begin(), bias_);

    return (float)response;
  }

  std::string LinearFeatureResponseSVM::ToString() const
  {
    std::stringstream s;
    s << "LinearFeatureResponse()";

    return s.str();
  }


        } } }
