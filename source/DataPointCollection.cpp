#include "DataPointCollection.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>



namespace Slither
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


  std::unique_ptr<DataPointCollection> DataPointCollection::Load(const std::string &filename)
  {
    std::ifstream file(filename);
    if(!file.is_open())
      return std::unique_ptr<DataPointCollection>();

    std::vector<std::vector<float>> values;
    std::vector<int> labels;
    std::string line;

    while(std::getline(file, line))
    {
      std::vector<std::string> tokens;
      tokenize(line, tokens, " \t");
      if(tokens.size() < 2) continue;
      labels.push_back(static_cast<int>(to_float(tokens[0])));
      std::vector<float> row(tokens.size()-1);
      for(size_t i=1;i<tokens.size();++i)
        row[i-1] = to_float(tokens[i]);
      values.push_back(row);
    }

    auto result = std::unique_ptr<DataPointCollection>(new DataPointCollection());
    if(values.empty())
      return result;
    int rows = values.size();
    int cols = values[0].size();
    result->reserve(rows, cols);
    for(int i=0;i<rows;i++)
      for(int j=0;j<cols;j++)
        result->putValue(values[i][j], labels[i], i, j);
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

  std::unique_ptr<DataPointCollection> DataPointCollection::Generate2dGrid(
    std::pair<float, float> rangeX, int nStepsX,
    std::pair<float, float> rangeY, int nStepsY)
  {

    if (rangeX.first >= rangeX.second)
      throw std::runtime_error("Invalid x-axis range.");
    if (rangeY.first >= rangeY.second)
      throw std::runtime_error("Invalid y-axis range.");

    std::unique_ptr<DataPointCollection> result =  std::unique_ptr<DataPointCollection>(new DataPointCollection());

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
  std::unique_ptr<DataPointCollection> DataPointCollection::Generate1dGrid(std::pair<float, float> range, int nSteps)
  {
    if (range.first >= range.second)
      throw std::runtime_error("Invalid range.");

    std::unique_ptr<DataPointCollection> result =  std::unique_ptr<DataPointCollection>(new DataPointCollection());

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


    auto viewMat = dataMat.leftCols(dimension);
    float min = viewMat.minCoeff();
    float max = viewMat.maxCoeff();
    return std::pair<float, float>(min, max);
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

    for (int i = 0; i < dataMat.cols(); i++)
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
}
