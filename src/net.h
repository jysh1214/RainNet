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
    void initRandomWieght();
    void predict(float* input);
    void train(float* input);

    bool training;
    float* input;
    float* target;
    std::vector<Layer*> layers;
    bool loadWieght = false;
};

#endif