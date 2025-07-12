#include "FeatureResponseFunctions.h"

#include <cmath>

#include <sstream>
#include <numeric>
#include <svm.h>
#include "DataPointCollection.h"
#include "Random.h"

namespace Slither
{
  AxisAlignedFeatureResponse AxisAlignedFeatureResponse ::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
  {
    return AxisAlignedFeatureResponse(random.Next(0, 2));
  }

  float AxisAlignedFeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const
  {
    const DataPointCollection& concreteData = (DataPointCollection&)(data);
    return concreteData.GetDataPoint((int) sampleIndex)[axis_];
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
    Eigen::RowVectorXf rowMat = concreteData.GetDataPoint(index);
    return dx_ * rowMat(0) + dy_ * rowMat(1);
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
    //std::cout<<" Selecting : ";
    for(int i=0;i<numBloks;i++)
    {
      int indx = random.Next(0,dims);
      //std::cout<<indx<<" ";
      vIndex.push_back(indx);
    }
    //std::cout<<std::endl;

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
    int numBloks = random.Next (5, 50);
    for(int i=0;i<numBloks;i++)
    {
      int idx = random.Next (0,NN_DIM);
      vIndex.push_back (idx);
      //vIndex.push_back (idx+NN_DIM);
    }

  }

void LinearFeatureResponseSVM::GenerateMaskHypercolumnStatistics(Random &random, std::vector<int> &vIndex, int dims,
                                                         bool root_node)
  {

    //Discarding LBP and position to check
    int numBloks = random.Next (5, 50);
    for(int i=0;i<numBloks;i++)
    {
      int idx = random.Next (0,NN_DIM);
      vIndex.push_back (idx);
      vIndex.push_back (idx+NN_DIM);
    }

  }
  void LinearFeatureResponseSVM::GenerateMaskHypercolumn2LrStatistics(Random &random, std::vector<int> &vIndex, int dims,
                                                                   bool root_node)
  {

    //Discarding LBP and position to check
    int numBloks = random.Next (5, 100);
    for(int i=0;i<numBloks;i++)
    {
      int idx = random.Next (0,NN_DIM*2);
      vIndex.push_back (idx);
      vIndex.push_back (idx+NN_DIM*2);
    }

  }


  void LinearFeatureResponseSVM::GenerateMaskHypercolumnStatisticsAndLocation(Random &random, std::vector<int> &vIndex, int dims,
                                                         bool root_node)
  {

    //Discarding LBP and position to check
    int numBloks = random.Next (5, 50);
    for(int i=0;i<numBloks;i++)
    {
      int idx = random.Next (0,NN_DIM);
      vIndex.push_back (idx);
      vIndex.push_back (idx+NN_DIM);
    }


    //Location
    //Big Assumption, Location is the last 2 dimensions (X,Y)
    bool loc_choice = random.NextDouble() > 0.5;
    if(loc_choice)
    {
      vIndex.push_back(dims-2);
      vIndex.push_back(dims-1);
    }


  }


  void LinearFeatureResponseSVM::GenerateMaskHypercolumnStatisticsColorAndLocation(Random &random, std::vector<int> &vIndex, int dims,
                                                                              bool root_node)
  {

    //Discarding LBP and position to check
    int numBloks = random.Next (5, 50);
    for(int i=0;i<numBloks;i++)
    {
      int idx = random.Next (0,NN_DIM);
      vIndex.push_back (idx);
      vIndex.push_back (idx+NN_DIM);
    }


    //Location
    //Big Assumption, Location is the last 4-2 dimensions (X,Y)
    bool loc_choice = random.NextDouble() > 0.5;
    if(loc_choice)
    {
      vIndex.push_back(dims-5);
      vIndex.push_back(dims-4);
    }

    //Color
    //Bigg assumption - last 3 dimensions
    bool color_choice = random.NextDouble() > 0.5;
    if(color_choice)
    {
      vIndex.push_back(dims-3);
      vIndex.push_back(dims-2);
      vIndex.push_back(dims-1);
    }


  }


  void LinearFeatureResponseSVM::GenerateMaskHypercolumnStatisticsLBPColorAndLocation(Random& random, std::vector<int>& vIndex, int dims , bool root_node)
  {


        int numBloks = random.Next (5, 50);
        for(int i=0;i<numBloks;i++)
        {
          int idx = random.Next (0,NN_DIM);
          vIndex.push_back (idx);
          vIndex.push_back (idx+NN_DIM);
        }

        static const int LBP_DIM=58;

        //LBP CHOICE. If selected, select all LBP
        bool lbp_choice = random.NextDouble() > 0.5;
        if(lbp_choice)
        {
          for(int j=NN_DIM*2;j<(NN_DIM*2 + LBP_DIM);j++)
            vIndex.push_back(j);
        }


        //Location
        //Big Assumption, Location is the last 4-2 dimensions (X,Y)
        bool loc_choice = random.NextDouble() > 0.5;
        if(loc_choice)
        {
          vIndex.push_back(dims-5);
          vIndex.push_back(dims-4);
        }

        //Color
        //Bigg assumption - last 3 dimensions
        bool color_choice = random.NextDouble() > 0.5;
        if(color_choice)
        {
          vIndex.push_back(dims-3);
          vIndex.push_back(dims-2);
          vIndex.push_back(dims-1);
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
      case hypercolumn_loc:GenerateMaskHypercolumnStatisticsAndLocation (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
            break;
      case hypercolumn2:GenerateMaskHypercolumn2LrStatistics (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
            break;
      case hypercolumn_loc_color:GenerateMaskHypercolumnStatisticsColorAndLocation (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
            break;
      case hypercolumn_lbp_loc_color:GenerateMaskHypercolumnStatisticsLBPColorAndLocation (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
            break;
      default: std::cout<<"Using unknown mask function. Re-check parameters"<<std::endl;
    }

    std::sort(lr.vIndex_.begin(), lr.vIndex_.end());
    lr.vIndex_.erase( unique( lr.vIndex_.begin(), lr.vIndex_.end() ), lr.vIndex_.end() );

    lr.nWeights_ = lr.vIndex_.size();
    lr.vWeights_.assign(lr.nWeights_, 0.f);
    lr.bias_ = 0.f;

    svm_parameter param{};
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.C = svm_c;
    param.eps = 0.01;
    param.cache_size = 100;
    param.nr_weight = 0;

    int nsamples = i1 - i0;
    svm_problem prob{};
    prob.l = nsamples;
    std::vector<double> y(nsamples);
    std::vector<std::vector<svm_node>> xstorage(nsamples);
    std::vector<svm_node*> x(nsamples);

    for(int n=0; n<nsamples; ++n)
    {
      int idx = dataIndices[n+i0];
      Eigen::RowVectorXf row = concreteData.GetDataPoint(idx);
      xstorage[n].resize(lr.nWeights_ + 1);
      for(int j=0;j<lr.nWeights_;++j)
      {
        xstorage[n][j].index = j+1;
        xstorage[n][j].value = row(lr.vIndex_[j]);
      }
      xstorage[n][lr.nWeights_].index = -1;
      x[n] = xstorage[n].data();
      y[n] = concreteData.GetIntegerLabel(idx);
    }

    prob.x = x.data();
    prob.y = y.data();

    svm_model* model = svm_train(&prob, &param);
    for(int s=0; s<model->l; ++s)
    {
      double alpha = model->sv_coef[0][s];
      svm_node* sv = model->SV[s];
      for(int j=0;j<lr.nWeights_;++j)
        lr.vWeights_[j] += alpha * sv[j].value;
    }
    lr.bias_ = -model->rho[0];

    svm_free_and_destroy_model(&model);

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
    Eigen::RowVectorXf rowMat = concreteData.GetDataPoint(index);

    Eigen::VectorXf features(nWeights_);
    Eigen::Map<const Eigen::VectorXf> weights(vWeights_.data(), nWeights_);

    for (int i = 0; i < nWeights_; i++)
       features(i) = rowMat(vIndex_[i]);

    float response = features.dot(weights) + bias_;
    
    return response;
  }

  std::string LinearFeatureResponseSVM::ToString() const
  {
    std::stringstream s;
    s << "LinearFeatureResponse()";

    return s.str();
  }

}