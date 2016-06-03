#pragma once

// This file defines types used to illustrate the use of the decision forest
// library in simple multi-class classification task (2D data points).

#include <stdexcept>
#include <algorithm>

#include "Graphics.h"

#include "Sherwood.h"

#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "DataPointCollection.h"
#include "Classification.h"
#include "PlotCanvas.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  template<class F>
  class IFeatureResponseFactory
  {
  public:
    virtual F CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)=0;
  };

  class LinearFeatureFactory: public IFeatureResponseFactory<LinearFeatureResponse>
  {
  public:
      LinearFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node);
  };


  class AxisAlignedFeatureResponseFactory : public IFeatureResponseFactory<AxisAlignedFeatureResponse>
  {
  public:
      AxisAlignedFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node);
  };

  class LinearFeatureSVMFactory: public IFeatureResponseFactory<LinearFeatureResponseSVM>
  {
  public:
      LinearFeatureResponseSVM CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node);
  };

  template<class F>
  class ClassificationTrainingContext : public ITrainingContext<F,HistogramAggregator> // where F:IFeatureResponse
  {
  private:
    int nClasses_;



      IFeatureResponseFactory<F>* featureFactory_;

  public:
      IGType igType;
    ClassificationTrainingContext(int nClasses, IFeatureResponseFactory<F>* featureFactory)
    {
      nClasses_ = nClasses;
      featureFactory_ = featureFactory;
    }

  private:
    // Implementation of ITrainingContext
    F GetRandomFeature(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, FeatureMaskType featureMask, bool root_node)
    {
      return featureFactory_->CreateRandom(random, data, dataIndices,i0,i1,svm_c, featureMask, root_node);
    }

    HistogramAggregator GetStatisticsAggregator()
    {
      return HistogramAggregator(nClasses_);
    }

      double ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
      {
        //std::cout<<"Selected IG : "<<igType<<std::endl;
        switch(igType)
        {
          case ig_shannon:
            return ComputeInformationGainShannon(allStatistics, leftStatistics, rightStatistics);
                break;
          case ig_gini:
            return ComputeInformationGainGINI(allStatistics, leftStatistics, rightStatistics);
                break;
          default:
            printf("ERROR: Unknown IG type\n");
                exit(0);
        };
      }

      double ComputeInformationGainShannon(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
      {
        unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
        if (nTotalSamples <= 1)
          return 0.0;

        double entropyBefore = allStatistics.Entropy();
        double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.Entropy() + rightStatistics.SampleCount() * rightStatistics.Entropy()) / nTotalSamples;
        return entropyBefore - entropyAfter;
      }

      double ComputeInformationGainGINI(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
      {
        unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
        if (nTotalSamples <= 1)
          return 0.0;

        double entropyBefore = allStatistics.EntropyGINI();
        double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.EntropyGINI() + rightStatistics.SampleCount() * rightStatistics.EntropyGINI()) / nTotalSamples;
        return entropyBefore - entropyAfter;
      }

      double ComputeInformationGainReweighted(const HistogramAggregator& global, const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
      {
        // TODO: Bad!!!!
        unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
        if (nTotalSamples <= 1)
          return 0.0;

        double entropyBefore = allStatistics.Entropy(global.GetBins(), global.SampleCount());
        double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.Entropy(global.GetBins(), global.SampleCount()) + rightStatistics.SampleCount() * rightStatistics.Entropy(global.GetBins(), global.SampleCount())) / nTotalSamples;
        return entropyBefore - entropyAfter;
      }

    bool ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain)
    {
      return gain < 0.01;
    }
  };

  template<class F>
  class ClassificationDemo
  {
    static const PixelBgr UnlabelledDataPointColor;

  public:
    static std::auto_ptr<Forest<F, HistogramAggregator> > Train (
      const DataPointCollection& trainingData,
      IFeatureResponseFactory<F>* featureFactory,
      const TrainingParameters& trainingParameters ) // where F : IFeatureResponse
    {
      if (trainingData.HasLabels() == false)
        throw std::runtime_error("Training data points must be labelled.");
      if (trainingData.HasTargetValues() == true)
        throw std::runtime_error("Training data points should not have target values.");

      std::cout << "Running training..." << std::endl;

      Random random;

      ClassificationTrainingContext<F> classificationContext(trainingData.CountClasses(), featureFactory);
      classificationContext.igType = trainingParameters.igType;


      std::auto_ptr<Forest<F, HistogramAggregator> > forest 
        = ParallelForestTrainer<F, HistogramAggregator>::TrainForest (
        random, trainingParameters, classificationContext, trainingData );

      return forest;
    }


    static std::auto_ptr<Forest<F, HistogramAggregator> > TrainSingle (
            const DataPointCollection& trainingData,
            IFeatureResponseFactory<F>* featureFactory,
            const TrainingParameters& trainingParameters ) // where F : IFeatureResponse
    {
      if (trainingData.HasLabels() == false)
        throw std::runtime_error("Training data points must be labelled.");
      if (trainingData.HasTargetValues() == true)
        throw std::runtime_error("Training data points should not have target values.");

      std::cout << "Running training..." << std::endl;
      //trainingData.showMat();

      Random random;

      ClassificationTrainingContext<F> classificationContext(trainingData.CountClasses(), featureFactory);
      classificationContext.igType = trainingParameters.igType;

      std::auto_ptr<Forest<F, HistogramAggregator> > forest
              = ForestTrainer<F, HistogramAggregator>::TrainForest (
                      random, trainingParameters, classificationContext, trainingData );

      return forest;
    }



    static std::auto_ptr<Forest<F, HistogramAggregator> > TrainParallel (
            const DataPointCollection& trainingData,
            IFeatureResponseFactory<F>* featureFactory,
            const TrainingParameters& trainingParameters ) // where F : IFeatureResponse
    {
      if (trainingData.HasLabels() == false)
        throw std::runtime_error("Training data points must be labelled.");
      if (trainingData.HasTargetValues() == true)
        throw std::runtime_error("Training data points should not have target values.");

      std::cout << "Running training..." << std::endl;
      //trainingData.showMat();

      Random random;

      ClassificationTrainingContext<F> classificationContext(trainingData.CountClasses(), featureFactory);
      classificationContext.igType = trainingParameters.igType;

      std::auto_ptr<Forest<F, HistogramAggregator> > forest
              = ForestTrainer<F, HistogramAggregator>::TrainForestParallel (
                      random, trainingParameters, classificationContext, trainingData );

      return forest;
    }


    /// <summary>
    /// Apply a trained forest to some test data.
    /// </summary>
    /// <typeparam name="F">Type of split function</typeparam>
    /// <param name="forest">Trained forest</param>
    /// <param name="testData">Test data</param>
    /// <returns>An array of class distributions, one per test data point</returns>
    static void Test(const Forest<F, HistogramAggregator>& forest, const DataPointCollection& testData, std::vector<HistogramAggregator>& distributions) // where F : IFeatureResponse
    {
      int nClasses = forest.GetTree(0).GetNode(0).TrainingDataStatistics.BinCount();

      std::vector<std::vector<int> > leafIndicesPerTree;
      forest.Apply(testData, leafIndicesPerTree);

      std::vector<HistogramAggregator> result(testData.Count());
      int correctCount = 0;

      std::vector<float> road_results;
      std::cout<<" Applying "<<forest.TreeCount()<<" number of trees : "<<std::endl;

      for (int i = 0; i < testData.Count(); i++)
      {
        // Aggregate statistics for this sample over all leaf nodes reached
        result[i] = HistogramAggregator(nClasses);
        for (int t = 0; t < forest.TreeCount(); t++)
        {
          int leafIndex = leafIndicesPerTree[t][i];
          result[i].Aggregate(forest.GetTree(t).GetNode(leafIndex).TrainingDataStatistics);

        }
        //std::cout<<testData.GetIntegerLabel(i)<<"|"<<result[i].FindTallestBinIndex()<<" | "<<result[i].GetProbability(result[i].FindTallestBinIndex())<<std::endl;
        correctCount += (testData.GetIntegerLabel(i) == result[i].FindTallestBinIndex());
        road_results.push_back(result[i].GetProbability(1));

      }

      std::cout<<" Score : "<<correctCount<<" / "<<testData.Count()<<std::endl;

      cv::Mat bla = testData.reconstructPredictions(0, road_results);

      distributions  = result;

      //return result;
    }






    /// <summary>
    /// Apply a trained forest to some test data.
    /// </summary>
    /// <typeparam name="F">Type of split function</typeparam>
    /// <param name="forest">Trained forest</param>
    /// <param name="testData">Test data</param>
    /// <returns>An array of class distributions, one per test data point</returns>
    static void TestAndPredictIm(const Forest<F, HistogramAggregator>& forest, const DataPointCollection& testData, std::vector<HistogramAggregator>& distributions, std::string folder_prediction) // where F : IFeatureResponse
    {
      int nClasses = forest.GetTree(0).GetNode(0).TrainingDataStatistics.BinCount();

      std::vector<std::vector<int> > leafIndicesPerTree;
      forest.Apply(testData, leafIndicesPerTree);

      std::vector<HistogramAggregator> result(testData.Count());
      int correctCount = 0;

      std::vector<float> road_results;
      std::cout<<" Applying "<<forest.TreeCount()<<" number of trees : "<<std::endl;



      for (int i = 0; i < testData.Count(); i++)
      {
        // Aggregate statistics for this sample over all leaf nodes reached
        result[i] = HistogramAggregator(nClasses);
        for (int t = 0; t < forest.TreeCount(); t++)
        {
          int leafIndex = leafIndicesPerTree[t][i];
          result[i].Aggregate(forest.GetTree(t).GetNode(leafIndex).TrainingDataStatistics);

        }
        //std::cout<<testData.GetIntegerLabel(i)<<"|"<<result[i].FindTallestBinIndex()<<" | "<<result[i].GetProbability(result[i].FindTallestBinIndex())<<std::endl;
        correctCount += (testData.GetIntegerLabel(i) == result[i].FindTallestBinIndex());
        road_results.push_back(result[i].GetProbability(1));

      }

      std::cout<<" Score : "<<correctCount<<" / "<<testData.Count()<<std::endl;



      distributions  = result;

      for(int r=0;r<testData.filenames.size();r++)
      {
        std::string out_name = folder_prediction+testData.filenames[r];
        std::cout<<"Reconstructing predictions for image : "<<r<<" --> "<<out_name<<std::endl;
        cv::Mat bla = testData.reconstructPredictions(r, road_results)*255;
        cv::Mat m_out;
        bla.convertTo(m_out, CV_8U);
        //std::cout<<bla;
        //cv::imshow("bla",bla);
        //cv::waitKey();
        cv::imwrite(out_name,bla);

      }

      //return result;
    }
  };

  template<class F>
  const PixelBgr ClassificationDemo<F>::UnlabelledDataPointColor = PixelBgr::FromArgb(192, 192, 192);
} } }
