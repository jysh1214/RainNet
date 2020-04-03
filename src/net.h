#ifndef NET_H
#define NET_H

#include "layer.h"
#include "loss_function.h"

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

struct Net {
  void init();
  void mallocLayerMemory();
  void createRandomWeight();
  void createRandomBias();
  void predict(tensor *input);
  void train(tensor *input, size_t epoch);

  bool training;
  float learningRate;
  float error;
  std::string lossFunction;
  float (*LossFunction)(tensor *, tensor *);

  tensor *input;
  tensor *target;
  std::vector<Layer *> layers;
  bool loadweight = false;
};

#endif
