#include "DataPointCollection.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  std::istream & getline_(std::istream & in, std::string & out)
  {
    out.clear();
    char c;
    while(in.get(c).good())
    {
      if(c == '\n')
      {
        c = in.peek();
        if(in.good())
        {
          if(c == '\r')
          {
            in.ignore();
          }
        }
        break;
      }
      out.append(1,c);
    }
    return in;
  }
            /*

  std::auto_ptr<DataPointCollection> DataPointCollection::Load(std::istream& r, int dataDimension, DataDescriptor::e descriptor)
  {
    bool bHasTargetValues = (descriptor & DataDescriptor::HasTargetValues) == DataDescriptor::HasTargetValues;
    bool bHasClassLabels = (descriptor & DataDescriptor::HasClassLabels) == DataDescriptor::HasClassLabels;

    std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());
    result->dimension_ = dataDimension;


    unsigned int elementsPerLine = (bHasClassLabels ? 1 : 0) + dataDimension + (bHasTargetValues ? 1 : 0);

    std::string line;
    while (r.good())
    {
      getline_(r, line);

      if(r.fail())
        throw std::runtime_error("Failed to read line.");

      std::vector<std::string> elements;
      tokenize(line, elements, "\t"); // split tab-delimited line

      if (elements.size() != elementsPerLine)
        throw std::runtime_error("Encountered line with unexpected number of elements.");

      int index = 0;

      if (bHasClassLabels)
      {
        if(elements[index]!="")
        {
          if (result->labelIndices_.find(elements[index])==result->labelIndices_.end())
            result->labelIndices_.insert(std::pair<std::string, int>(elements[index], result->labelIndices_.size()));

          result->labels_.push_back(result->labelIndices_[elements[index++]]);
          result->uniqueClasses_.insert(result->labelIndices_[elements[index]]); //TODO : CHECK in c++
        }
        else
        {
          // cast necessary in g++ because std::vector<int>::push_back() takes a reference
          result->labels_.push_back((int)(DataPointCollection::UnknownClassLabel));
          result->uniqueClasses_.insert((int)(DataPointCollection::UnknownClassLabel)); //TODO:Check in c++
          index++;
        }
      }

      cv::Mat rowMat = cv::Mat(1,dataDimension,CV_32FC1);

      for (int i = 0; i < dataDimension; i++)
      {

        float x = to_float(elements[index++]);
        rowMat.at<float>(0,i) =x;
//        result->data_.push_back(x);
      }

      if (bHasTargetValues)
      {
        float t = to_float(elements[index++]);
        result->targets_.push_back(t);
      }

      result->dataMat.push_back(rowMat);
    }

    for(int i = 0;i<result->labels_.size();i++)
      std::cout<<result->labels_[i]<<" ";
    std::cout<<std::endl<<result->labels_.size();


    return result;
  }*/

  std::auto_ptr<DataPointCollection> DataPointCollection::Load(const std::string &filename)
  {
    std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());

    cv::Ptr<cv::ml::TrainData> all_data;

    const cv::String bla = cv::String();
    try
    {
      all_data = cv::ml::TrainData::loadFromCSV(filename,0,0,-1,bla,'\t','?');
    }
    catch (cv::Exception & e)
    {
      std::cout<<"An Exception while reading has occurred: "<<e.what()<<std::endl;
    }


    cv::Mat responses = all_data->getResponses().t();

    const float* p = responses.ptr<float>(0);
    result->labels_ = std::vector<int> (p, p + responses.cols);


    //HACK - REmove it before it messes up the data
    for(int i=0;i<result->labels_.size();i++)
      result->labels_[i] = result->labels_[i]>0?1:0;


    result->uniqueClasses_ = std::set<int> (result->labels_.begin(), result->labels_.end());



    result->dataMat = cv::Mat(all_data->getSamples());
    result->dimension_ = all_data->getNVars();
    result->targets_.clear();


    all_data.release();


    return result;




  }

  /// <summary>
  /// NOTWORKING : Generate a 2D dataset with data points distributed in a grid pattern.
  /// Intended for generating visualization images.
  /// </summary>
  /// <param name="rangeX">x-axis range</param>
  /// <param name="nStepsX">Number of grid points in x direction</param>
  /// <param name="rangeY">y-axis range</param>
  /// <param name="nStepsY">Number of grid points in y direction</param>
  /// <returns>A new DataPointCollection</returns>

  std::auto_ptr<DataPointCollection> DataPointCollection::Generate2dGrid(
    std::pair<float, float> rangeX, int nStepsX,
    std::pair<float, float> rangeY, int nStepsY)
  {

    if (rangeX.first >= rangeX.second)
      throw std::runtime_error("Invalid x-axis range.");
    if (rangeY.first >= rangeY.second)
      throw std::runtime_error("Invalid y-axis range.");

    std::auto_ptr<DataPointCollection> result =  std::auto_ptr<DataPointCollection>(new DataPointCollection());

    result->dimension_ = 2;

    float stepX = (rangeX.second - rangeX.first) / nStepsX;
    float stepY = (rangeY.second - rangeY.first) / nStepsY;

    for (int j = 0; j < nStepsY; j++)
      for (int i = 0; i < nStepsX; i++)
      {
        //RPG-Removed
        //result->data_.push_back(rangeX.first + i * stepX);
        //result->data_.push_back(rangeY.first + j * stepY);
      }

      return result;
  }

  /// <summary>
  /// NOTWORKING - Generate a 1D dataset containing a given number of data points
  /// distributed at regular intervals within a given range. Intended for
  /// generating visualization images.
  /// </summary>
  /// <param name="range">Range</param>
  /// <param name="nStepsX">Number of grid points</param>
  /// <returns>A new DataPointCollection</returns>
  std::auto_ptr<DataPointCollection> DataPointCollection::Generate1dGrid(std::pair<float, float> range, int nSteps)
  {
    if (range.first >= range.second)
      throw std::runtime_error("Invalid range.");

    std::auto_ptr<DataPointCollection> result =  std::auto_ptr<DataPointCollection>(new DataPointCollection());

    result->dimension_ = 1;

    float step = (range.second - range.first) / nSteps;

    for (int i = 0; i < nSteps; i++)
    {
      //REMOVED - RPG
      //result->data_.push_back(range.first + i * step);
    }
    return result;
  }


  /// <summary>
  /// Get the data range in the specified data dimension.
  /// </summary>
  /// <param name="dimension">The dimension over which to compute min and max</param>
  /// <returns>A tuple containing min and max over the specified dimension of the data</returns>
  std::pair<float, float> DataPointCollection::GetRange(int dimension) const
  {
    if (Count() < 0)
      throw std::runtime_error("Insufficient data to compute range.");

    if (dimension < 0 || dimension>dimension_)
      throw std::runtime_error("Insufficient data to compute range.");

/*    float min = data_[0 + dimension], max = data_[0 + dimension];

    for (int i = 0; i < (int)(data_.size())/dimension_; i++)
    {
      if (data_[i*dimension_ +  dimension] < min)
        min = data_[i*dimension_ +  dimension];
      else if (data_[i*dimension_ +  dimension] > max)
        max = data_[i*dimension_ +  dimension];
    }
*/


    cv::Mat viewMat = dataMat.colRange(0, dimension);
    double min, max;
    cv::minMaxLoc(viewMat, &min, &max);
    return std::pair<float, float>((float)min, (float)max);
  }

  /// <summary>
  /// Get the range of target values (or raise an exception if these data
  /// do not have associated target values).
  /// </summary>
  /// <returns>A tuple containing the min and max target value for the data</returns>
  std::pair<float, float> DataPointCollection::GetTargetRange() const
  {
    if (!HasTargetValues())
      throw std::runtime_error("Data points do not have target values.");
    if (Count() < 0)
      throw std::runtime_error("Insufficient data to compute range.");

    float min = targets_[0], max = targets_[0];

    for (int i = 0; i < dataMat.cols; i++)
    {
      if (targets_[i] < min)
        min = targets_[i];
      else if (targets_[i] > max)
        max = targets_[i];
    }

    return std::pair<float, float>(min, max);
  }

  void tokenize(
    const std::string& str,
    std::vector<std::string>& tokens,
    const std::string& delimiters)
  {
    tokens.clear();

    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0); // skip initial delimeters

    for(std::string::size_type i=0; i<(lastPos==std::string::npos?str.length()+1:lastPos); i++)
      tokens.push_back("");

    std::string::size_type pos = str.find_first_of(delimiters, lastPos); // first "non-delimiter"   

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
      tokens.push_back(str.substr(lastPos, pos - lastPos)); // found token, add to vector 
      lastPos = str.find_first_not_of(delimiters, pos);     // skip delimiters

      if(pos!=std::string::npos)
        for(std::string::size_type i=pos+1; i<(lastPos==std::string::npos?str.length()+1:lastPos); i++)
          tokens.push_back("");

      pos = str.find_first_of(delimiters, lastPos);         // find next "non-delimiter"
    }
  }

  float to_float(const std::string& s)
  {
    std::istringstream ss(s);  
    float x;
    ss >> x;
    if(ss.fail())
      throw std::runtime_error("Failed to interpret number as floating point.");
    return x;
  }
} } }
