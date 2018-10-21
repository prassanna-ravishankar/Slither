#include "Classification.h"

#include "Features.h"

namespace Slither
{
    AxisAlignedFeature AxisAlignedFeatureFactory::CreateRandom(Random& random)
    {
        return AxisAlignedFeature::CreateRandom(random);
    }

	LinearFeature2d LinearFeatureFactory::CreateRandom(Random& random)
    {
        return LinearFeature2d::CreateRandom(random);
    }
}
