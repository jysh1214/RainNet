#ifndef NET_H
#define NET_H

#include "layer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>

struct Net
{
    void init();
    void createRandomWeight();
    void predict(float* input);
    void train(float* input, size_t epoch);

    bool training;
    float learningRate;
    float error;

    float* input;
    float* target;
    std::vector<Layer*> layers;
    bool loadweight = false;
};

#endif