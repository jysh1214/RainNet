#ifndef NET_H
#define NET_H

#include "layer.h"

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
    void free();

    float* input;
    std::vector<Layer*> layers;
    bool loadWieght = false;
};

#endif