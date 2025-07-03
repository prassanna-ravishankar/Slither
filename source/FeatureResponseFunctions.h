#pragma once

// This file defines some IFeatureResponse implementations used by the example code in
// Classification.h, DensityEstimation.h, etc. Note we represent IFeatureResponse
// instances using simple structs so that all tree data can be stored
// contiguously in a linear array.

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>


#include "Sherwood.h"
#include <boost/serialization/serialization.hpp>
#include "../lib/external/json.hpp"

namespace cvml = cv::ml;

namespace Slither
{
  class Random;


#define WITH_FISHER true
  #define FISHER_CLUSTERS 16
  #define HYPER_MACHINE_DIM 64
  #define HYPER_LBP_DIM 59
  #define HYPER_LOCATION_DIM 2
  #define HYPER_MACHINE_PAIRS (HYPER_MACHINE_DIM*FISHER_CLUSTERS) //64*16 ->1024
  #define HYPER_LBP_PAIRS (HYPER_LBP_DIM*FISHER_CLUSTERS) //59*64 -> 3776
  #define HYPERFISHER_MACHINE_DIM (2*HYPER_MACHINE_PAIRS) //2*1024  --> 2048
  #define HYPERFISHER_LBP_DIM (2*HYPER_LBP_PAIRS) //2*3776 -->
  #define HYPER_FISHER_DIM (HYPER_MACHINE_DIM+HYPER_LBP_DIM+HYPER_LOCATION_DIM) //2048 + 2*3776
  #define BLOK_SIZE_SUPERPIXEL 2
  #define NN_DIM 64
  #define SUPERPIXEL_STATISTICS 2
  #define NN_FULL_DIMS (NN_DIM*SUPERPIXEL_STATISTICS)

  /// <summary>
  /// A feature that orders data points using one of their coordinates,
  /// i.e. by projecting them onto a coordinate axis.
  /// </summary>
  class AxisAlignedFeatureResponse
  {
    int axis_;

  public:
    AxisAlignedFeatureResponse()
    {
      axis_ = -1;
    }

    /// <summary>
    /// Create an AxisAlignedFeatureResponse instance for the specified axis.
    /// </summary>
    /// <param name="axis">The zero-based index of the axis.</param>
    AxisAlignedFeatureResponse(int axis)
    {
      axis_ = axis;
    }

    /// <summary>
    /// Create an AxisAlignedFeatureResponse instance with a random choice of axis.
    /// </summary>
    /// <param name="randomNumberGenerator">A random number generator.</param>
    /// <returns>A new AxisAlignedFeatureResponse instance.</returns>
    static AxisAlignedFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node);

    int Axis() const
    {
      return axis_;
    }

    // IFeatureResponse implementation
    float GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const;

    std::string ToString() const;
  };

  /// <summary>
  /// A feature that orders data points using a linear combination of their
  /// coordinates, i.e. by projecting them onto a given direction vector.
  /// </summary>
  class LinearFeatureResponse2d
  {
    float dx_, dy_;

  public:
    LinearFeatureResponse2d()
    {
      dx_ = 0.0;
      dy_ = 0.0;
    }

    /// <summary>
    /// Create a LinearFeatureResponse2d instance for the specified direction vector.
    /// </summary>
    /// <param name="dx">The first element of the direction vector.</param>
    /// <param name="dx">The second element of the direction vector.</param> 
    LinearFeatureResponse2d(float dx, float dy)
    {
      dx_ = dx; dy_ = dy;
    }

    /// <summary>
    /// Create a LinearFeatureResponse2d instance with a random direction vector.
    /// </summary>
    /// <param name="randomNumberGenerator">A random number generator.</param>
    /// <returns>A new LinearFeatureResponse2d instance.</returns>
    static LinearFeatureResponse2d CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node);

    // IFeatureResponse implementation
    float GetResponse(const IDataPointCollection& data, unsigned int index) const;

    std::string ToString()  const;
  };


  class LinearFeatureResponse
  {
  protected:
      std::vector<float> vWeights_;
      int		dimensions_;
      float	bias_;
      int		nIndex_;


  public:
      //std::string ;
      LinearFeatureResponse():
              dimensions_(-1),
              bias_(0.0f)
      {

      }

      /// <summary>
      /// Create a LinearFeatureResponse instance for the specified direction vector.
      /// </summary>
      /// <param name="dx">The first element of the direction vector.</param>
      /// <param name="dx">The second element of the direction vector.</param>
      LinearFeatureResponse(float* pWeights, const int dimensions)
      {

        vWeights_ = std::vector<float>(pWeights, pWeights+sizeof pWeights/sizeof pWeights[0]);
        dimensions_ = dimensions;
      }

      /// <summary>
      /// Create a LinearFeatureResponse2d instance with a random direction vector.
      /// </summary>
      /// <param name="randomNumberGenerator">A random number generator.</param>
      /// <returns>A new LinearFeatureResponse2d instance.</returns>
      static LinearFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node);

      // IFeatureResponse implementation
      float GetResponse(const IDataPointCollection& data, unsigned int index) const;

      std::string ToString()  const;
  };


  class LinearFeatureResponseSVM
  {
      friend class boost::serialization::access;
  protected:
      std::vector<int> vIndex_;
      std::vector<float> vWeights_;
      int		dimensions_;
      float	bias_;
      //int		nIndex_;
      //cv::Ptr<cvml::SVM> svm;
      int nWeights_;

  public:
      LinearFeatureResponseSVM():
              dimensions_(-1),
              bias_(0.0f)
      {
          dimensions_ = 10;
          vWeights_.resize(dimensions_,1);
          for(int i=0;i<dimensions_;i++)
            vIndex_.push_back(i);
          nWeights_ = vWeights_.size();
        //m_param_filename = "/home/prassanna/Development/Code3/Parameters/parametersTaskManager2.ini";
          //svm = cvml::SVM::create();
          //svm->setType(cvml::SVM::C_SVC);
          //svm->setKernel(cvml::SVM::LINEAR);
          //svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 1000, 0.01));

      }

      /// <summary>
      /// Create a LinearFeatureResponse instance for the specified direction vector.
      /// </summary>
      /// <param name="dx">The first element of the direction vector.</param>
      /// <param name="dx">The second element of the direction vector.</param>
      LinearFeatureResponseSVM(float* pWeights, const int dimensions)
      {

        //vWeights_ = std::vector<float>(pWeights, pWeights+sizeof pWeights/sizeof pWeights[0]);
        dimensions_ = dimensions;
      }
      static LinearFeatureResponseSVM CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node);
      static void GenerateMask(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskFisher(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskLBP(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskHypercolumn(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskHypercolumnStatistics(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskHypercolumn2LrStatistics(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskHypercolumnStatisticsAndLocation(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskHypercolumnStatisticsColorAndLocation(Random& random, std::vector<int>& vIndex, int dims , bool root_node);
      static void GenerateMaskHypercolumnStatisticsLBPColorAndLocation(Random& random, std::vector<int>& vIndex, int dims , bool root_node);


      float GetResponse(const IDataPointCollection &data, unsigned int index) const;
      std::string ToString()  const;

      static void Serialize(LinearFeatureResponseSVM lr, std::ostream& o)
      {
          o.write((const char*)(&lr.dimensions_), sizeof(lr.dimensions_));
          o.write((const char*)(&lr.nWeights_), sizeof(lr.nWeights_));
          o.write((const char*)(&lr.vIndex_), sizeof(int)*lr.vIndex_.size());
          o.write(reinterpret_cast<const char*>(&lr.vIndex_[0]), lr.vIndex_.size()*sizeof(int));
          o.write(reinterpret_cast<const char*>(&lr.vWeights_[0]), lr.vWeights_.size()*sizeof(float));

      }

      static LinearFeatureResponseSVM Deserialize(std::istream& i)
      {
          LinearFeatureResponseSVM lr;
          i.read((char*)(&lr.dimensions_), sizeof(lr.dimensions_));
          i.read((char*)(&lr.nWeights_), sizeof(lr.nWeights_));
          i.read((char*)(&lr.vIndex_), sizeof(lr.vIndex_));
          i.read(reinterpret_cast<char*>(&lr.vIndex_[0]), lr.nWeights_*sizeof(int));
          i.read(reinterpret_cast<char*>(&lr.vWeights_[0]), lr.nWeights_*sizeof(float));


          return lr;


      }

      //FOR BOOST SERIALIZATION
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
      {
          ar & vIndex_;
          ar & vWeights_;
          ar & dimensions_;
          ar & bias_;
          ar & nWeights_;
      }

      template<class Archive>
      void serializeBoost(Archive & ar)
      {
          ar & vIndex_;
          ar & vWeights_;
          ar & dimensions_;
          ar & bias_;
          ar & nWeights_;
      }

      template<class Archive>
      void deserializeBoost(Archive & ar)
      {
          ar & vIndex_;
          ar & vWeights_;
          ar & dimensions_;
          ar & bias_;
          ar & nWeights_;
      }

      // BEGIN JSON SERIALIZATION (Modern replacement)
      
      /// <summary>
      /// Serialize LinearFeatureResponseSVM to JSON (modern replacement for Boost serialization).
      /// </summary>
      void serializeJson(nlohmann::json& j) const
      {
          j["v_index"] = vIndex_;
          j["v_weights"] = vWeights_;
          j["dimensions"] = dimensions_;
          j["bias"] = bias_;
          j["n_weights"] = nWeights_;
      }
      
      /// <summary>
      /// Deserialize LinearFeatureResponseSVM from JSON (modern replacement for Boost serialization).
      /// </summary>
      void deserializeJson(const nlohmann::json& j)
      {
          vIndex_ = j["v_index"];
          vWeights_ = j["v_weights"];
          dimensions_ = j["dimensions"];
          bias_ = j["bias"];
          nWeights_ = j["n_weights"];
      }
      
      //END JSON SERIALIZATION

      /*void write(cv::FileStorage& fs) const                        //Write serialization for this class
      {
          cv::Mat_<int> indexMat (vIndex_,true);
          fs<<"fr_Index"<<indexMat;
          //std::cout<<indexMat.rows<<"  "<<indexMat.cols<<std::endl;
          std::string str = svm->isTrained()?"True":"False";
          fs<<"fr_svm_trained"<<(int)svm->isTrained();
          if(svm->isTrained())
          {
              fs<<"fr_svm"<<"[";
              this->svm->write(fs);
              fs<<"]";
          }
          fs<<"fr_dimensions" << dimensions_;

      }*/


      /*void read(cv::FileNode& fs)
      {
          cv::Mat_<int> indexMat;
          fs["fr_Index"]>>indexMat;
          vIndex_.clear();
          if(indexMat.cols>0)
              vIndex_.resize(indexMat.cols,0);
          for(int i = 0;i<indexMat.cols;i++)
              vIndex_[i] =indexMat(0,i);

          int trained = (int)fs["fr_svm_trained"];
          if(trained) {
              cv::FileNode fn = fs["fr_svm"];
              this->svm->read(fn);

          }

          fs["fr_dimensions"]>>dimensions_;
      }*/
  };



}
