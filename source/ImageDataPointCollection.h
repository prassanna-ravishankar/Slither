//
// Created by prassanna on 10/23/18.
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>


#include "Sherwood.h"
#include <set>
namespace cvml = cv::ml;



namespace Slither
{
    /// <summary>
    /// Used to describe the expected format of the lines of a data file (used
    /// in ImageDataPointCollection::Load()).
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
    class ImageDataPointCollection: public IDataPointCollection
    {
        //std::vector<float> data_;
        std::vector<std::string> filenames;
        int dimension_;
        std::set<int> uniqueClasses_;


        // only for classified data...
        std::vector<int> labels_;

        //std::map<std::string, int> labelIndices_; // map string labels to integers

        // only for regression problems...
        std::vector<float> targets_;

    public:
        static const int UnknownClassLabel = -1;

        /// <summary>
        /// Load a collection of data from a tab-delimited file with one data point
        /// per line. The data may optionally have associated with class labels
        /// (first element on line) and/or target values (last element on line).
        /// </summary>
        /// <param name="path">Path of file to be read.</param>
        /// <param name="bHasClassLabels">Are the data associated with class labels?</param>
        /// <param name="dataDimension">Dimension of the data (excluding class labels and target values).</param>
        /// <param name="bHasTargetValues">Are the data associated with target values.</param>
        //static  std::unique_ptr<ImageDataPointCollection> Load(std::istream& r, int dataDimension, DataDescriptor::e descriptor);

        static  std::unique_ptr<ImageDataPointCollection> Load(const std::string &filename);


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
            return filenames.size();
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
        /// <returns>Filename of the image</returns>
        const std::string GetDataPoint(int i) const
        {
            return filenames[i];
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

