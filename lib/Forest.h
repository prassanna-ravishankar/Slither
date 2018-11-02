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
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>


namespace Slither
{
  /// <summary>
  /// A decision forest, i.e. a collection of decision trees.
  /// </summary>
  template<class F, class S>
  class Forest // where F:IFeatureResponse where S:IStatisticsAggregator<S>
  {
    static const char* binaryFileHeader_;

    std::vector< Tree<F,S>* > trees_;

  public:
    typedef typename std::vector< Tree<F,S>* >::size_type TreeIndex;

    ~Forest()
    {
      for(TreeIndex t=0; t<trees_.size(); t++)
        delete trees_[t];
    }

    /// <summary>
    /// Add another tree to the forest.
    /// </summary>
    /// <param name="path">The tree.</param>
    void AddTree(std::unique_ptr<Tree<F,S> >& tree)
    {
      tree->CheckValid();

      trees_.push_back(tree.get());
      tree.release();
    }

    /// <summary>
    /// Deserialize a forest from a file.
    /// </summary>
    /// <param name="path">The file path.</param>
    /// <returns>The forest.</returns>
    static std::unique_ptr<Forest<F, S> > DeserializeBoost(const std::string& path)
    {
      std::ifstream i(path.c_str(), std::ios_base::binary);

      std::unique_ptr<Forest<F, S> > forest = std::unique_ptr<Forest<F, S> >(new Forest<F,S>());

      int treecount;
      boost::archive::text_iarchive ar(i);
      ar & treecount;
      std::unique_ptr<Tree<F,S> > tree;
      for(int t=0; t<treecount; t++)
      {
        tree =  Tree<F, S>::deserializeTree(ar);
        std::cout<<"Tree : "<<t<<std::endl;

        forest->trees_.push_back(tree.release());
      }
      return forest;
    }


    /// <summary>
    /// Serialize the forest to file.
    /// </summary>
    /// <param name="path">The file path.</param>
    void SerializeBoost(const std::string& path)
    {
      std::ofstream o(path.c_str(), std::ios_base::binary);
      boost::archive::text_oarchive ar(o);
      int treeCount = TreeCount();

      //First tree counts
      ar & treeCount;

      //Then Tree & decision levels
      for(int t=0; t<TreeCount(); t++)
      {
        GetTree(t).serializeTree(ar);
      }

      o.close();

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
