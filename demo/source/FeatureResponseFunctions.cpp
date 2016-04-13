#include "FeatureResponseFunctions.h"

#include <cmath>

#include <sstream>

#include "DataPointCollection.h"
#include "Random.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  AxisAlignedFeatureResponse AxisAlignedFeatureResponse ::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node)
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
  LinearFeatureResponse2d LinearFeatureResponse2d::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node)
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

} } }
