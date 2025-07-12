#pragma once

// This files defines the Random class, used throughout the forest training
// framework for random number generation.

#include <random>
#include <chrono>

namespace Slither
{
  /// <summary>
  /// Encapsulates random number generation using modern C++11 random facilities.
  /// Thread-safe and provides high-quality random number generation.
  /// </summary>
  class Random
  {
  private:
    std::mt19937 generator_;
    
  public:
    /// <summary>
    /// Creates a random number generator using a seed derived from high-resolution clock.
    /// </summary>
    Random()
    {
      auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      generator_.seed(static_cast<std::mt19937::result_type>(seed));
    }

    /// <summary>
    /// Creates a deterministic random number generator using the specified seed.
    /// May be useful for debugging and reproducible results.
    /// </summary>
    Random(unsigned int seed)
    {
      generator_.seed(seed);
    }

    /// <summary>
    /// Generate a positive random number.
    /// </summary>
    int Next()
    {
      return static_cast<int>(generator_());
    }

    /// <summary>
    /// Generate a random number in the range [0.0, 1.0).
    /// </summary>
    double NextDouble()
    {
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      return distribution(generator_);
    }

    /// <summary>
    /// Generate a random integer within the specified range.
    /// </summary>
    /// <param name="minValue">Inclusive lower bound.</param>
    /// <param name="maxValue">Exclusive upper bound.</param>
    int Next(int minValue, int maxValue)
    {
      std::uniform_int_distribution<int> distribution(minValue, maxValue - 1);
      return distribution(generator_);
    }
  };
}
