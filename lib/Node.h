#pragma once

// This file defines the Node data structure, which is used to represent one node
// in a DecisionTree.
#include <nlohmann/json.hpp>

namespace Slither
{
  // Default serialization functions used for serializing Feature and
  // StatisticsAggregator types. If your implementations are not simple
  // value types then use explicit template instantiation to override.
  template<class T>
  void Serialize_(std::ostream& o, const T& t)
  {
    o.write((const char*)(&t), sizeof(T));
  }

  template<class T>
  void Deserialize_(std::istream& o, T& t)
  {
    o.read((char*)(&t), sizeof(T));
  }

  /// <summary>
  /// One node in a decision tree.
  /// </summary>

  // *** NB We implement Nodes and their constituent IFeatureResponse and
  // IStatisticsAggregrator implementations using value types so as to
  // avoid the need for multiple objects to be allocated seperately on the
  // garbage collected heap - instead all the data associated with a tree can
  // be stored within a single, contiguous block of memory. This can
  // increase performance by (i) increasing spatial locality of reference
  // (and thus cache utilization) and (ii) decreasing the load on the .NET
  // memory manager.

  template<class F, class S>
  struct Node // where F : IFeatureResponse where S: IStatisticsAggregator<S>
  {
    // NB Null nodes (i.e. uninitialized nodes corresponding to the bottom of
    // a tree branch for which training was terminated before maximum tree
    // depth) have bIsleaf==bIsSplit_==false. Please see IsSplit(),
    // IsLeaf(), IsNull(), below.

    bool bIsLeaf_;
    bool bIsSplit_;

    // The weak learner associated with a decision node is defined by a
    // feature and an associated threshold. These values are only valid
    // for split nodes.
    F Feature;

    float Threshold;

    // NB We store training data statistics for all nodes, including
    // decision nodes - this way we can prune the tree subsequent to
    // training.
    S TrainingDataStatistics;


    void InitializeLeaf(S trainingDataStatistics)
    {
      Feature = F();
      Threshold = 0.0f;
      bIsLeaf_ = true;
      bIsSplit_ = false;
      TrainingDataStatistics = trainingDataStatistics.DeepClone();
    }

    void InitializeSplit(F feature, float threshold, S trainingDataStatistics)
    {
      bIsLeaf_ = false;
      bIsSplit_ = true;
      Feature = feature;
      Threshold = threshold;
      TrainingDataStatistics = trainingDataStatistics.DeepClone();
    }

    Node()
    {
      // Nodes are created null by default
      bIsLeaf_ = false;
      bIsSplit_ = false;
      Feature = F();
      Threshold = 0.0f;
      TrainingDataStatistics = S();
    }

   /* void Serialize(std::ostream& o) const
    {
      Serialize_(o, bIsLeaf_);
      Serialize_(o, bIsSplit_);
      Serialize_(o, Feature);
      Serialize_(o, Threshold);
      Serialize_(o, TrainingDataStatistics);
    }

    void Deserialize(std::istream& i)
    {
      Deserialize_(i, bIsLeaf_);
      Deserialize_(i, bIsSplit_);
      Deserialize_(i, Feature);
      Deserialize_(i, Threshold);
      Deserialize_(i, TrainingDataStatistics);
    }*/

    //FOR BOOST SERIALIZATION
//    template<class Archive>
//    void serialize(Archive & ar, const unsigned int version)
//    {
//
//      ar & bIsLeaf_;
//      //std::cout<<"L ";
//      ar & bIsSplit_;
//      //std::cout<<"S ";
//      ar & Threshold;
//      //std::cout<<"T ";
//      TrainingDataStatistics = S();
//      ar & TrainingDataStatistics;
//      //std::cout<<"S ";
//      ar & Feature;
//      //std::cout<<"F ";
//      std::cout<<std::endl;
//    }

    // BEGIN JSON SERIALIZATION (Modern replacement)
    
    /// <summary>
    /// Serialize node to JSON (modern replacement for Boost serialization).
    /// </summary>
    void serializeJson(nlohmann::json& j) const
    {
      j["is_leaf"] = bIsLeaf_;
      j["is_split"] = bIsSplit_;
      j["threshold"] = Threshold;
      
      // Serialize training data statistics (assuming it has JSON serialization)
      nlohmann::json stats_json;
      TrainingDataStatistics.serializeJson(stats_json);
      j["training_data_statistics"] = stats_json;
      
      // Serialize feature (assuming it has JSON serialization)
      nlohmann::json feature_json;
      Feature.serializeJson(feature_json);
      j["feature"] = feature_json;
    }
    
    /// <summary>
    /// Deserialize node from JSON (modern replacement for Boost serialization).
    /// </summary>
    void deserializeJson(const nlohmann::json& j)
    {
      bIsLeaf_ = j["is_leaf"];
      bIsSplit_ = j["is_split"];
      Threshold = j["threshold"];
      
      // Deserialize training data statistics
      TrainingDataStatistics.deserializeJson(j["training_data_statistics"]);
      
      // Deserialize feature
      Feature.deserializeJson(j["feature"]);
    }
    
    //END JSON SERIALIZATION

    /// <summary>
    /// Is this a decision node, i.e. a node with an associated weak learner
    /// and child nodes?
    /// </summary>
    bool IsSplit() const { return bIsSplit_ && !bIsLeaf_; }

    /// <summary>
    /// Is this a leaf node, i.e. a node with no associated weak learner
    /// or child nodes?
    /// </summary>
    bool IsLeaf() const { return bIsLeaf_ && !bIsSplit_; }

    /// <summary>
    /// Is this an uninitialized node (corresponding to the bottom of
    // a tree branch for which training was terminated before maximum tree
    // depth or a not-yet-trained node)?
    /// </summary>
    bool IsNull() const { return !bIsLeaf_ && !bIsSplit_; }
  };
}
