#include "Classification.h"

#include "FeatureResponseFunctions.h"

namespace Slither
{
    AxisAlignedFeatureResponse AxisAlignedFeatureResponseFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
    {
      return AxisAlignedFeatureResponse::CreateRandom(random, data, dataIndices,i0,i1,svm_c,featureMask, root_node);
    }

    /*
    LinearFeatureResponse2d LinearFeatureFactory::CreateRandom(Random& random)
    {
      return LinearFeatureResponse2d::CreateRandom(random);
    }*/

    LinearFeatureResponse LinearFeatureFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
    {
      return LinearFeatureResponse::CreateRandom(random, data, dataIndices,i0,i1,svm_c, featureMask, root_node);
    }

    LinearFeatureResponseSVM LinearFeatureSVMFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
    {
      return LinearFeatureResponseSVM::CreateRandom(random, data, dataIndices,i0,i1,svm_c,featureMask, root_node);
    }

    PatchLinearFeatureResponseSVM PatchLinearFeatureSVMFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
    {
        return PatchLinearFeatureResponseSVM::CreateRandom(random, data, dataIndices,i0,i1,svm_c,featureMask, root_node);
    }
}

