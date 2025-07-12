#pragma once

// Main header file for Sherwood's object oriented framework for
// decision forest inference.

#include "Random.h"

#include "Forest.h"
#include "Tree.h"
#include "Node.h"

#include "ForestTrainer.h"
#ifdef WITH_OPENMP
#include "ParallelForestTrainer.h"
#endif

#include "Interfaces.h"
