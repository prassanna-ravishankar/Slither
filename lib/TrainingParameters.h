#pragma once

// This file defines the ParallelParallelTreeTraininer class, which is responsible for
// creating new Tree instances by learning from training data.

#include <assert.h>

#include <vector>
#include <string>
#include <algorithm>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  enum IGType { ig_unknown=-1, ig_shannon, ig_reweighted, ig_gini };



  /// <summary>
  /// Decision tree training parameters.
  /// </summary>
  struct TrainingParameters
  {
    TrainingParameters()
    {
      // Some sane defaults will need to be changed per application.
      NumberOfTrees = 1;
      NumberOfCandidateFeatures = 10;
      NumberOfCandidateThresholdsPerFeature = 10;
      MaxDecisionLevels = 5;
      Verbose = false;
      svm_c = 0.5;
      igType = ig_gini;
      featureMask = standard;
      maxThreads = 4;
    }

    int NumberOfTrees;
    int NumberOfCandidateFeatures;
    unsigned int NumberOfCandidateThresholdsPerFeature;
    int MaxDecisionLevels;
    bool Verbose;
    float svm_c;
    IGType  igType;
    FeatureMaskType featureMask;
      int maxThreads;


  };
} } }