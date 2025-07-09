#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

#include <Eigen/Dense>

#include "Sherwood.h"
#include <set>



namespace Slither
{
  /// <summary>
  /// Used to describe the expected format of the lines of a data file (used
  /// in DataPointCollection::Load()).
  /// </summary>
  class DataDescriptor
  {
  public:
    enum e
    {
      Unadorned = 0x0,
      HasClassLabels = 0x1,
      HasTargetValues = 0x2
    };
  };

  /// <summary>
  /// A collection of data points, each represented by a float[] and (optionally)
  /// associated with a string class label and/or a float target value.
  /// </summary>
  class DataPointCollection: public IDataPointCollection
  {
    //std::vector<float> data_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dataMat;
    int dimension_;
    std::set<int> uniqueClasses_;


      // only for classified data...
    std::vector<int> labels_;

    //std::map<std::string, int> labelIndices_; // map string labels to integers

    // only for regression problems...
    std::vector<float> targets_;

  public:
    static const int UnknownClassLabel = -1;

    Eigen::VectorXf scaleRow(const Eigen::VectorXf& rowVec, float& bias, float& factor )
    {
      bias = rowVec.minCoeff();
      factor = rowVec.maxCoeff();
      if(factor == 0)
        return rowVec;
      return ((rowVec.array() - bias) / factor - 0.5f) * 2.0f;
    }

    void scaleData(std::vector<float>& biases, std::vector<float>&divisors)
    {
      Eigen::MatrixXf target_mat(dataMat.rows(), dataMat.cols());
      biases.assign(dataMat.cols(), 0);
      divisors.assign(dataMat.cols(), 1);
      for(int i=0;i<dataMat.cols();i++)
      {
        Eigen::VectorXf colvec = dataMat.col(i);
        Eigen::VectorXf processed = scaleRow(colvec,biases[i],divisors[i]);
        target_mat.col(i) = processed;
      }

      dataMat =  target_mat;
    }

    void doScaleData(const Eigen::VectorXf& biases, const Eigen::VectorXf& divisors)
    {
      dataMat = (dataMat.rowwise() - biases.transpose()).array().rowwise() / divisors.transpose().array();
    }

    bool reserve(int H, int W)
    {
      dataMat = Eigen::MatrixXf::Zero(H, W);
      dimension_ = W;
      labels_.resize(H,0);
      return true;
    }

    bool putValue(float value,int label, int h,int w)
    {
      dataMat(h,w) = value;
      labels_[h] = label;
      uniqueClasses_.insert(label);
      //std::cout<<"{";for(std::set<int>::iterator iter=uniqueClasses_.begin(); iter!=uniqueClasses_.end();++iter) { std::cout<<(*iter);}std::cout<<"}"<<std::endl;


        //std::cout<<"Putting Value : "<<value<<" lbl : "<<label<<std::endl;
        return true;
    }

    /// <summary>
    /// Load a collection of data from a tab-delimited file with one data point
    /// per line. The data may optionally have associated with class labels
    /// (first element on line) and/or target values (last element on line).
    /// </summary>
    /// <param name="path">Path of file to be read.</param>
    /// <param name="bHasClassLabels">Are the data associated with class labels?</param>
    /// <param name="dataDimension">Dimension of the data (excluding class labels and target values).</param>
    /// <param name="bHasTargetValues">Are the data associated with target values.</param>
    //static  std::unique_ptr<DataPointCollection> Load(std::istream& r, int dataDimension, DataDescriptor::e descriptor);

     static  std::unique_ptr<DataPointCollection> Load(const std::string &filename);

    /// <summary>
    /// Generate a 2D dataset with data points distributed in a grid pattern.
    /// Intended for generating visualization images.
    /// </summary>
    /// <param name="rangeX">x-axis range</param>
    /// <param name="nStepsX">Number of grid points in x direction</param>
    /// <param name="rangeY">y-axis range</param>
    /// <param name="nStepsY">Number of grid points in y direction</param>
    /// <returns>A new DataPointCollection</returns>
    static  std::unique_ptr<DataPointCollection> Generate2dGrid(
      std::pair<float, float> rangeX, int nStepsX,
      std::pair<float, float> rangeY, int nStepsY);

    /// <summary>
    /// Generate a 1D dataset containing a given number of data points
    /// distributed at regular intervals within a given range. Intended for
    /// generating visualization images.
    /// </summary>
    /// <param name="range">Range</param>
    /// <param name="nStepsX">Number of grid points</param>
    /// <returns>A new DataPointCollection</returns>
    static std::unique_ptr<DataPointCollection> Generate1dGrid(std::pair<float, float> range, int nSteps);

    /// <summary>
    /// Do these data have class labels?
    /// </summary>
    bool HasLabels() const
    {
      return labels_.size() != 0;
    }

    /// <summary>
    /// How many unique class labels are there?
    /// </summary>
    /// <returns>The number of unique class labels</returns>
    int CountClasses() const
    {
      if (!HasLabels())
        throw std::runtime_error("Unlabelled data.");

      //return labelIndices_.size();
      return uniqueClasses_.size();
    }

    /// <summary>
    /// Do these data have target values (e.g. for regression)?
    /// </summary>
    bool HasTargetValues() const
    {
      return targets_.size() != 0;
    }

    /// <summary>
    /// Count the data points in this collection.
    /// </summary>
    /// <returns>The number of data points</returns>
    unsigned int Count() const
    {
      return dataMat.rows();
    }

    void showMat() const
    {
      std::cout<<dataMat<<std::endl;
      std::cout<<dataMat.size()<<std::endl;
    }

    /// <summary>
    /// Get the data range in the specified data dimension.
    /// </summary>
    /// <param name="dimension">The dimension over which to compute min and max</param>
    /// <returns>A tuple containing min and max over the specified dimension of the data</returns>
    std::pair<float, float> GetRange(int dimension) const;

    /// <summary>
    /// Get the range of target values (or raise an exception if these data
    /// do not have associated target values).
    /// </summary>
    /// <returns>A tuple containing the min and max target value for the data</returns>
    std::pair<float, float> GetTargetRange() const;

    /// <summary>
    /// The dimensionality of the data (excluding optional target values).
    /// </summary>
    int Dimensions() const
    {
      return dimension_;
    }

    /// <summary>
    /// Get the specified data point.
    /// </summary>
    /// <param name="i">Zero-based data point index.</param>
    /// <returns>Pointer to the first element of the data point.</returns>
    /*const float* GetDataPoint(int i) const
    {
      return &data_[i*dimension_];
    }*/

    /// <summary>
    /// Get the specified data point.
    /// </summary>
    /// <param name="i">Zero-based data point index.</param>
    /// <returns>Row values of the first element of the data point.</returns>
    Eigen::RowVectorXf GetDataPoint(int i) const
    {
      return dataMat.row(i);
    }

    /// <summary>
    /// Get data point as Eigen vector for efficient linear algebra operations.
    /// </summary>
    /// <param name="i">Zero-based data point index.</param>
    /// <returns>Eigen row vector view of the data point.</returns>
    Eigen::Map<const Eigen::RowVectorXf> GetDataPointEigen(int i) const
    {
      return Eigen::Map<const Eigen::RowVectorXf>(dataMat.row(i).data(), dataMat.cols());
    }

    /// <summary>
    /// Get entire data matrix as Eigen matrix for batch operations.
    /// </summary>
    /// <returns>Eigen matrix view of all data points.</returns>
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& GetDataMatrixEigen() const
    {
      return dataMat;
    }





    /// <summary>
    /// Get the class label for the specified data point (or raise an
    /// exception if these data points do not have associated labels).
    /// </summary>
    /// <param name="i">Zero-based data point index</param>
    /// <returns>A zero-based integer class label.</returns>
    int GetIntegerLabel(int i) const
    {
      if (!HasLabels())
        throw std::runtime_error("Data have no associated class labels.");
      return labels_[i]; // may throw an exception if index is out of range
    }

    /// <summary>
    /// Get the target value for the specified data point (or raise an
    /// exception if these data points do not have associated target values).
    /// </summary>
    /// <param name="i">Zero-based data point index.</param>
    /// <returns>The target value.</returns>
    float GetTarget(int i) const
    {
      if (!HasTargetValues())
        throw std::runtime_error("Data have no associated target values.");

      return targets_[i]; // may throw an exception if index is out of range
    }
  };

  // A couple of file parsing utilities, exposed here for testing only.

  // Split a delimited line into constituent elements.
  void tokenize(
    const std::string& str,
    std::vector<std::string>& tokens,
    const std::string& delimiters = " " );

  // Convert a std::string to a float (or raise an exception).
  float to_float(const std::string& s);
}
