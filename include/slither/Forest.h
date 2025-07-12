#pragma once

// This file declares the Forest class, which is used to represent forests
// of decisions trees.

#include <memory>
#include <stdexcept>
#include <fstream>
#include <istream>
#include <iostream>
#include <vector>

#include "ProgressStream.h"

#include "Interfaces.h"
#include "Tree.h"
#include <nlohmann/json.hpp>


namespace Slither
{
  /// <summary>
  /// A decision forest, i.e. a collection of decision trees.
  /// </summary>
  template<class F, class S>
  class Forest // where F:IFeatureResponse where S:IStatisticsAggregator<S>
  {
    static const char* binaryFileHeader_;

    std::vector< std::unique_ptr<Tree<F,S>> > trees_;

  public:
    typedef typename std::vector< std::unique_ptr<Tree<F,S>> >::size_type TreeIndex;

    // Destructor is now automatic with unique_ptr - no manual cleanup needed
    ~Forest() = default;

    /// <summary>
    /// Add another tree to the forest.
    /// </summary>
    /// <param name="tree">The tree to add (ownership transferred).</param>
    void AddTree(std::unique_ptr<Tree<F,S>> tree)
    {
      tree->CheckValid();
      trees_.push_back(std::move(tree));
    }

    /// <summary>
    /// Deserialize a forest from a file.
    /// </summary>
    /// <param name="path">The file path.</param>
    /// <returns>The forest.</returns>

    /// <summary>
    /// Serialize the forest to JSON file (modern replacement for Boost serialization).
    /// </summary>
    /// <param name="path">The file path.</param>
    void SerializeJson(const std::string& path)
    {
      nlohmann::json j;
      j["format_version"] = "1.0";
      j["tree_count"] = TreeCount();
      j["trees"] = nlohmann::json::array();
      
      for(int t = 0; t < TreeCount(); t++)
      {
        nlohmann::json tree_json;
        GetTree(t).serializeJson(tree_json);
        j["trees"].push_back(tree_json);
      }
      
      std::ofstream o(path);
      o << j.dump(2); // Pretty-printed with 2-space indentation
      o.close();
    }

    /// <summary>
    /// Deserialize a forest from JSON file (modern replacement for Boost serialization).
    /// </summary>
    /// <param name="path">The file path.</param>
    /// <returns>The forest.</returns>
    static std::unique_ptr<Forest<F, S>> DeserializeJson(const std::string& path)
    {
      std::ifstream i(path);
      nlohmann::json j;
      i >> j;
      i.close();
      
      auto forest = std::make_unique<Forest<F, S>>();
      
      // Validate format version
      if(j.contains("format_version")) {
        std::string version = j["format_version"];
        if(version != "1.0") {
          throw std::runtime_error("Unsupported forest format version: " + version);
        }
      }
      
      int tree_count = j["tree_count"];
      const auto& trees_json = j["trees"];
      
      for(int t = 0; t < tree_count; t++)
      {
        auto tree = Tree<F, S>::deserializeJson(trees_json[t]);
        std::cout << "Loaded tree " << t << std::endl;
        forest->trees_.push_back(std::move(tree));
      }
      
      return forest;
    }


    /// <summary>
    /// Access the specified tree.
    /// </summary>
    /// <param name="index">A zero-based integer index.</param>
    /// <returns>The tree.</returns>
    const Tree<F,S>& GetTree(int index) const
    {
      return *trees_[index];
    }

    /// <summary>
    /// Access the specified tree.
    /// </summary>
    /// <param name="index">A zero-based integer index.</param>
    /// <returns>The tree.</returns>
    Tree<F,S>& GetTree(int index)
    {
      return *trees_[index];
    }


    /// <summary>
    /// How many trees in the forest?
    /// </summary>
    int TreeCount() const
    {
      return trees_.size();
    }

    /// <summary>
    /// Apply a forest of trees to a set of data points.
    /// </summary>
    /// <param name="data">The data points.</param>
    void Apply(
      const IDataPointCollection& data,
      std::vector<std::vector<int> >& leafNodeIndices,
      ProgressStream* progress=0 ) const
    {
      ProgressStream defaultProgressStream(std::cout, Interest);
      progress = (progress==0)?&defaultProgressStream:progress;

      leafNodeIndices.resize(TreeCount());

      for (int t = 0; t < TreeCount(); t++)
      {
        leafNodeIndices[t].resize(data.Count());

        (*progress)[Interest] << "\rApplying tree " << t << "...";
        trees_[t]->Apply(data, leafNodeIndices[t]);
      }

      (*progress)[Interest] << "\rApplied " << TreeCount() << " trees.        " << std::endl;
    }
  };

  template<class F, class S>
  const char* Forest<F,S>::binaryFileHeader_ = "MicrosoftResearch.Cambridge.Sherwood.Forest";
}
