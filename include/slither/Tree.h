#pragma once

// This file defines the Tree class, which is used to represent decision trees.

#include <assert.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "Interfaces.h"
#include "Node.h"
#include <nlohmann/json.hpp>



namespace Slither
{
  template<class F, class S> class TreeTrainer;
  template<class F, class S> class ParallelTreeTrainer;

  /// <summary>
  /// A decision tree, comprising multiple nodes.
  /// </summary>
  template<class F, class S> 
  class Tree // where F:IFeatureResponse where S:IStatisticsAggregator<S>
  {

    static const char* binaryFileHeader_;

    typedef typename std::vector<unsigned int>::size_type DataPointIndex;

    int decisionLevels_;

    std::vector<Node<F,S> > nodes_;

  public:
      Tree()
      {

      }
    // Implementation only
    Tree(int decisionLevels):decisionLevels_(decisionLevels)
    {
      if(decisionLevels<0)
        throw std::runtime_error("Tree can't have less than 0 decision levels.");

      /*
       * if(decisionLevels>19)
       *  throw std::runtime_error("Tree can't have more than 19 decision levels.");
       * Don't know why keep limit */


      // This full allocation of node storage may be wasteful of memory
      // if trees are unbalanced but is efficient otherwise. Because child
      // node indices can determined directly from the parent node's index
      // it isn't necessary to store parent-child references within the
      // nodes.
      nodes_.resize((1 << (decisionLevels + 1)) - 1);
    }

    std::vector<Node<F,S> > & GetNodes() { return nodes_;}

  public:
    /// <summary>
    /// Apply the decision tree to a collection of test data points.
    /// </summary>
    /// <param name="data">The test data.</param>
    /// <returns>An array of leaf node indices per data point.</returns>
    void Apply(const IDataPointCollection& data, std::vector<int>& leafNodeIndices)
    {
      CheckValid();

      leafNodeIndices.resize(data.Count()); // of leaf node reached per data point

      // Allocate temporary storage for data point indices and response values
      std::vector<unsigned int> dataIndices_(data.Count());
      for (unsigned int i = 0; i < data.Count(); i++)
        dataIndices_[i] = i;

      std::vector<float> responses_(data.Count());

      ApplyNode(0, data, dataIndices_, 0, data.Count(), leafNodeIndices, responses_);
    }
//




    // BEGIN JSON SERIALIZATION (Modern replacement)
    
    /// <summary>
    /// Serialize tree to JSON (modern replacement for Boost serialization).
    /// </summary>
    void serializeJson(nlohmann::json& j) const
    {
      j["decision_levels"] = decisionLevels_;
      j["node_count"] = NodeCount();
      j["nodes"] = nlohmann::json::array();
      
      for(int n = 0; n < NodeCount(); n++) {
        nlohmann::json node_json;
        nodes_[n].serializeJson(node_json);
        j["nodes"].push_back(node_json);
      }
    }
    
    /// <summary>
    /// Deserialize tree from JSON (modern replacement for Boost serialization).
    /// </summary>
    static std::unique_ptr<Tree<F,S>> deserializeJson(const nlohmann::json& j)
    {
      int decision_levels = j["decision_levels"];
      auto tree = std::make_unique<Tree<F, S>>(decision_levels);
      
      const auto& nodes_json = j["nodes"];
      for(int n = 0; n < tree->NodeCount(); n++) {
        tree->nodes_[n].deserializeJson(nodes_json[n]);
      }
      
      tree->CheckValid();
      return tree;
    }
    
    //END JSON SERIALIZATION

    /// <summary>
    /// The number of nodes in the tree, including decision, leaf, and null nodes.
    /// </summary>
    int NodeCount() const
    {
      return nodes_.size();
    }

    /// <summary>
    /// Return the specified tree node.
    /// </summary>
    /// <param name="index">A zero-based node index.</param>
    /// <returns>The node.</returns>
    const Node<F,S>& GetNode(int index) const
    {
      return nodes_[index];
    }

    /// <summary>
    /// Return the specified tree node.
    /// </summary>
    /// <param name="index">A zero-based node index.</param>
    /// <returns>The node.</returns>
    Node<F,S>& GetNode(int index)
    {
      return nodes_[index];
    }

    static DataPointIndex Partition(std::vector<float>& keys, std::vector<unsigned int>& values, DataPointIndex i0, DataPointIndex i1, float threshold)
    {
      assert(i1 > i0); // past-the-end element index must be greater than start element index.

      int i = (int)(i0);     // index of first element
      int j = int(i1 - 1); // index of last element

      while (i != j)
      {
        if (keys[i] >= threshold)
        {
          // Swap keys[i] with keys[j]
          float key = keys[i];
          unsigned int value = values[i];

          keys[i] = keys[j];
          values[i] = values[j];

          keys[j] = key;
          values[j] = value;

          j--;
        }
        else
        {
          i++;
        }
      }

      return keys[i] >= threshold ? i : i + 1;
    }

    void CheckValid() const
    {
      if(NodeCount()==0)
        throw std::runtime_error("Valid tree must have at least one node.");

      if(GetNode(0).IsNull()==true)
        throw std::runtime_error("A valid tree must have non-null root node.");

      CheckValidRecurse(0);
    }

  private:
    void CheckValidRecurse(int index, bool bHaveReachedLeaf=false) const
    {
      if (bHaveReachedLeaf==false && GetNode(index).IsLeaf())
      {
        // First time I have encountered a leaf node
        bHaveReachedLeaf = true;
      }
      else
      {
        if (bHaveReachedLeaf)
        {
          // Have encountered a leaf node already, this node had better be null
          if (GetNode(index).IsNull() == false)
            throw std::runtime_error("Valid tree must have all descendents of leaf nodes set as null nodes.");
        }
        else
        {
          // Have not encountered a leaf node yet, this node had better be a split node
          if (GetNode(index).IsSplit() == false)
            throw std::runtime_error("Valid tree must have all antecents of leaf nodes set as split nodes.");
        }
      }

      if (index >= (NodeCount() - 1) / 2)
      {
        // At maximum depth, this node had better be a leaf
        if (bHaveReachedLeaf == false)
          throw std::runtime_error("Valid tree must have all branches terminated by leaf nodes.");
      }
      else
      {
        CheckValidRecurse(2 * index + 1, bHaveReachedLeaf);
        CheckValidRecurse(2 * index + 2, bHaveReachedLeaf);
      }
    }

  public:
      int DecisionLevels()
      {
        return decisionLevels_;
      }
    static std::string GetPrettyPrintPrefix(int nodeIndex)
    {
      std::string prefix = nodeIndex > 0 ? (nodeIndex % 2 == 1 ? "|-o " : "+-o ") : "o ";
      for (int l = (nodeIndex - 1) / 2; l > 0; l = (l - 1) / 2)
        prefix = (l % 2 == 1 ? "| " : "  ") + prefix;
      return prefix;
    }

  private:
    void ApplyNode(
      int nodeIndex,
      const IDataPointCollection& data,
      std::vector<unsigned int>& dataIndices,
      int i0,
      int i1,
      std::vector<int>& leafNodeIndices,
      std::vector<float>& responses_)
    {
      assert(nodes_[nodeIndex].IsNull()==false);

      Node<F,S>& node = nodes_[nodeIndex];

      if (node.IsLeaf())
      {
        for (int i = i0; i < i1; i++)
          leafNodeIndices[dataIndices[i]] = nodeIndex;
        return;
      }

      if (i0 == i1)   // No samples left
        return;

      for (int i = i0; i < i1; i++)
        responses_[i] = node.Feature.GetResponse(data, dataIndices[i]);

      int ii = Partition(responses_, dataIndices, i0, i1, node.Threshold);

      // Recurse for child nodes.
      ApplyNode(nodeIndex * 2 + 1, data, dataIndices, i0, ii, leafNodeIndices, responses_);
      ApplyNode(nodeIndex * 2 + 2, data, dataIndices, ii, i1, leafNodeIndices, responses_);
    }



  };




  template<class F, class S>
  const char* Tree<F,S>::binaryFileHeader_ = "MicrosoftResearch.Cambridge.Sherwood.Tree";
}

