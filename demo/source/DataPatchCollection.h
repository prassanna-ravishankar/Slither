#pragma once

#include <string>
#include <vector>
#include <list>
#include <map>
#include <memory>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "Sherwood.h"
#include <set>
namespace cvml = cv::ml;

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{


            class Classifier {
            public:
                cv::Size input_geometry_;

                Classifier();

                Classifier(const std::string& model_file,
                           const std::string& trained_file);



                cv::Mat forwardPass(const cv::Mat &img,int layer_nr);
                std::vector<cv::Mat> forwardPassVector(const cv::Mat &img,int layer_nr);
                void setNetworkSources(const std::string& model_file,
                                       const std::string& trained_file,
                                       const std::string& mean_file,
                                       const std::string& label_file);
                void forwardPassCollection(const std::vector<cv::Mat> &images, int layer_nr, std::vector<int>&channels_required, std::vector<cv::Mat> &result_vector);
                void forwardPassSingle(const cv::Mat &img, int layer_nr, std::vector<int>&channels_required, cv::Mat &results);
                std::vector<float> calcHypercolumnPoint(const cv::Mat &img, int layer_nr, std::vector<int>&channels_required, cv::Point &location);
                std::vector<float> cachedHypercolumnPoint(int layer_nr, std::vector<int>&channels_required, cv::Point &location);
                cv::Mat getHypercolumnPoint(const cv::Mat &img, int layer_nr, std::vector<int>&channels_required, cv::Point &location);
                int getSuperpixelLabel(const cv::Mat &img_labels, cv::Point &location);
                std::vector<cv::Point> getSuperpixelPoints(const cv::Mat &img_labels, int lbl);
                cv::Mat getSuperpixelDescriptor(const cv::Mat &img, const cv::Mat &img_labels, int lbl, int layer_nr, std::vector<int> &channels_required);


            private:
                //void SetMean(const string& mean_file);


                void WrapInputLayer(std::vector<cv::Mat>* input_channels);

                void Preprocess(const cv::Mat& img,
                                std::vector<cv::Mat>* input_channels);

            private:

                std::shared_ptr<caffe::Net<float> > net_;
                int num_channels_;
                cv::Mat cached_image;
                cv::Mat cached_hypercolumn;
                //cv::Mat mean_;
                //std::vector<std::string> labels_;
                bool matIsEqual(const cv::Mat &Mat1, const cv::Mat &Mat2)
                {
                  if( Mat1.dims == Mat2.dims &&
                      Mat1.size == Mat2.size &&
                      Mat1.elemSize() == Mat2.elemSize())
                  {
                    if( Mat1.isContinuous() && Mat2.isContinuous())
                    {
                      return 0==memcmp( Mat1.ptr(), Mat2.ptr(), Mat1.total()*Mat1.elemSize());
                    }
                    else
                    {
                      const cv::Mat* arrays[] = {&Mat1, &Mat2, 0};
                      uchar* ptrs[2];
                      cv::NAryMatIterator it( arrays, ptrs, 2);
                      for(unsigned int p = 0; p < it.nplanes; p++, ++it)
                        if( 0!=memcmp( it.ptrs[0], it.ptrs[1], it.size*Mat1.elemSize()) )
                          return false;

                      return true;
                    }
                  }

                  return false;
                }
            };





  /// <summary>
  /// Used to describe the expected format of the lines of a data file (used
  /// in DataPointCollection::Load()).
  /// </summary>


  class Classifier;

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
  class DataPatchCollection: public IDataPointCollection
  {
    //std::vector<float> data_;
    //cv::Mat dataMat;
    std::vector<cv::Mat> images_;
    std::vector<cv::Mat> annotations_;
    std::vector<cv::Mat> superpixel_scale_1;
    std::vector<cv::Mat> superpixel_scale_2;
    std::vector<cv::Mat> superpixel_scale_3;
    std::vector<cv::Mat> hypercolumns_pixel;

    cv::Mat all_desc_pixels_;
    cv::Mat all_desc_superpixels_;
    cv::Mat all_desc_superpixels_;

    cv::Size  size_patches;

    std::auto_ptr<Classifier> net_;


      cv::Mat dataMat_; //Stores processed hypercolumns

    std::vector<std::string> filenames;
    std::vector<int> img_rows_;
    std::vector<int> img_cols_;
    std::vector<int> net_img_size_;

    //something for count
    int dataCount_;
    int dimension_;

    std::set<int> uniqueClasses_;

      // only for classified data...


    //std::map<std::string, int> labelIndices_; // map string labels to integers

    // only for regression problems...
    std::vector<float> targets_;

  public:
    static const int UnknownClassLabel = -1;
      std::string img_folder;
      std::string ann_folder;



      static  std::auto_ptr<DataPatchCollection> Create(std::string folder_img, std::string folder_ann, cv::Size size);

    /// <summary>
    /// Load a collection of data from a tab-delimited file with one data point
    /// per line. The data may optionally have associated with class labels
    /// (first element on line) and/or target values (last element on line).
    /// </summary>
    /// <param name="path">Path of file to be read.</param>
    /// <param name="bHasClassLabels">Are the data associated with class labels?</param>
    /// <param name="dataDimension">Dimension of the data (excluding class labels and target values).</param>
    /// <param name="bHasTargetValues">Are the data associated with target values.</param>
    //static  std::auto_ptr<DataPointCollection> Load(std::istream& r, int dataDimension, DataDescriptor::e descriptor);

     static  void Load(const std::string &filename, std::auto_ptr<DataPatchCollection> &patches);




    /// <summary>
    /// Do these data have class labels?
    /// </summary>
    bool HasLabels() const
    {
        //TODO :: Put some logic here
      return true;
    }

    /// <summary>
    /// How many unique class labels are there?
    /// </summary>
    /// <returns>The number of unique class labels</returns>
    int CountClasses() const
    {
        //ToDO :: Put some logic here to calculate unique classes
        return uniqueClasses_.size();
    }

    /// <summary>
    /// Do these data have target values (e.g. for regression)?
    /// </summary>
    bool HasTargetValues() const
    {
        //TODO :: Adapt for targets as well
      return false;
    }

    /// <summary>
    /// Count the data points in this collection.
    /// </summary>
    /// <returns>The number of data points</returns>
    unsigned int Count() const
    {

        return dataCount_;
      //return dataMat.rows;
    }

    void showMat() const
    {
      //std::cout<<dataMat<<std::endl;
      //std::cout<<dataMat.size()<<std::endl;
    }


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
    const cv::Mat GetDataPoint(int i, int superpixel_scale = 0) const //0 -> pixel descriptor , 1 -> superpixel_scale_1 and so on ...
    {

      //TODO :: Make better getdatapoint default function

      int k = findImage(i);
      int remainder = i-net_img_size_[k-1];
      int y = remainder / img_cols_[k];
      int x = remainder % img_cols_[k];
      int lbl_sup_1 = superpixel_scale_1[k].at<int>(y,x);
      int lbl_sup_2 = superpixel_scale_2[k].at<int>(y,x);
      int lbl_sup_3 = superpixel_scale_3[k].at<int>(y,x);
      //std::cout<<"Found in image  : "<<k<<" at pos ("<<x<<", "<<y<<")"<<std::endl;


      //ToDO :: Better return for this point

      return images_[k].clone();

    }


    const cv::Mat GetDataPoint(int i, int layer_nr, std::vector<int> &channels_required, bool superpixel_choice_1 =0, bool superpixel_choice_2=0, bool superpixel_choice=3)
    {

      int k = findImage(i);
      int diff = (k*size_patches.height*size_patches.width) - i;
      int x = diff % size_patches.width;
      int y = diff % size_patches.height;

      cv::Point pt(x,y);

      std::vector<cv::Mat> descriptors;
      cv::Mat desc_pxl = net_->getHypercolumnPoint(images_[k], layer_nr, channels_required, pt);
      if(superpixel_choice_1)
      {
        int lbl = net_->getSuperpixelLabel(superpixel_scale_1[k],pt);
        cv::Mat desc = net_->getSuperpixelDescriptor(images_[k], superpixel_scale_1[k], lbl, layer_nr, channels_required);
        descriptors.push_back(desc);
      }
      if(superpixel_choice_2)
      {
        int lbl = net_->getSuperpixelLabel(superpixel_scale_2[k],pt);
        cv::Mat desc = net_->getSuperpixelDescriptor(images_[k], superpixel_scale_2[k], lbl, layer_nr, channels_required);
        descriptors.push_back(desc);
      }
      if(superpixel_choice_2)
      {
        int lbl = net_->getSuperpixelLabel(superpixel_scale_3[k],pt);
        cv::Mat desc = net_->getSuperpixelDescriptor(images_[k], superpixel_scale_3[k], lbl, layer_nr, channels_required);
        descriptors.push_back(desc);
      }

      cv::Mat hypercolumn;
      cv::hconcat(descriptors, hypercolumn);

      return hypercolumn.clone();


    }

    const int findImage(int i) const
    {
      int size  = size_patches.height*size_patches.width;
      return (i/size);
    }


    cv::Ptr<cvml::TrainData> getTrainData()
    {

      //return cvml::TrainData::create(dataMat, cvml::ROW_SAMPLE, cv::Mat(labels_));

    }

    cv::Ptr<cvml::TrainData> getTrainDataWithMask(std::vector<int> mask_values, int start_row, int end_row)
    {
      /*
      cv::Mat colMat = dataMat.rowRange(start_row,end_row);
      //std::cout<<"--->"<<dataMat.rows<<std::endl;
      cv::Mat labelsMat = cv::Mat(labels_);
      cv::Mat reducedLabels = labelsMat.rowRange(start_row,end_row);
      return cvml::TrainData::create(colMat, cvml::ROW_SAMPLE, reducedLabels, mask_values);
       */
    }


    cv::Ptr<cvml::TrainData> getTrainDataWithMaskOrdered(std::vector<int> mask_values, int start_row, int end_row, unsigned int* indices)
    {
      /*cv::Mat colMat;
      //std::cout<<"--->"<<dataMat.rows<<std::endl;
      cv::Mat labelsMat = cv::Mat(labels_);
      cv::Mat reducedLabels;
      for(int i=start_row;i<end_row;i++)
      {
        int j = indices[i];
        colMat.push_back(dataMat.row(j));
        reducedLabels.push_back(labelsMat.row(j));
      }

      //std::cout<<colMat.size()<<std::endl;
      //std::cout<<reducedLabels.size()<<std::endl;

      return cvml::TrainData::create(colMat, cvml::ROW_SAMPLE, reducedLabels, mask_values);
       */
    }

      cv::Ptr<cvml::TrainData> getTrainDataWithMaskHypercolumn(std::vector<int> mask_values, int layer_nr, int start_row, int end_row, unsigned int* indices)
      {

        //TODO :: HOW TO GET SUPERPIXELS AS WELL, in the same structure?


        std::vector<cv::Mat> nn_feats;
        net_->forwardPassCollection(images_,layer_nr,mask_values,nn_feats);

        //std::cout<<nn_feats[0]<<std::endl<<nn_feats[0].size()<<"   "<<nn_feats[0].channels()<<std::endl;
        cv::Mat all_feats;


        for(int  i = 0;i<nn_feats.size();i++)
        {
          cv::Mat bla = nn_feats[i].reshape(1, nn_feats[i].rows*nn_feats[i].cols);
          all_feats.push_back(bla);
        }

        std::cout<<all_feats.size()<<std::endl;


      }





    /// <summary>
    /// Get the class label for the specified data point (or raise an
    /// exception if these data points do not have associated labels).
    /// </summary>
    /// <param name="i">Zero-based data point index</param>
    /// <returns>A zero-based integer class label.</returns>
    int GetIntegerLabel(int i) const
    {

        //TODO :: Call getDataPoint and then IntegerLabel

        return 1;
    }


      static void uniqueElements(const cv::Mat& input, std::set<int>& classes_)
      {
        for(int y=0;y<input.rows;y++)
          for(int x=0;x<input.cols;x++)
          {
              classes_.insert((int) input.at<uchar>(y,x));
          }
      }

  /*  /// <summary>
    /// Get the target value for the specified data point (or raise an
    /// exception if these data points do not have associated target values).
    /// </summary>
    /// <param name="i">Zero-based data point index.</param>
    /// <returns>The target value.</returns>
    float GetTarget(int i) const
    {

        //TODO :: Modify this to handle it organically
      if (!HasTargetValues())
        throw std::runtime_error("Data have no associated target values.");
    return 0;
    }*/
  };








} } }
