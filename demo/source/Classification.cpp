#include "Classification.h"

#include "FeatureResponseFunctions.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  AxisAlignedFeatureResponse AxisAlignedFeatureResponseFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c,bool root_node)
  {
    return AxisAlignedFeatureResponse::CreateRandom(random, data, dataIndices,i0,i1,svm_c, root_node);
  }


  LinearFeatureResponse2d LinearFeatureFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c,bool root_node)
  {
    return LinearFeatureResponse2d::CreateRandom(random, data, dataIndices,i0,i1,svm_c, root_node);
  }
} } }

