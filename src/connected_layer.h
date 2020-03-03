#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "layer.h"
#include "activation_fumction.h"

#include <stdlib.h>
#include <string>

class ConnectedLayer: public Layer
{
public:
    ConnectedLayer(size_t size, std::string activation);
    virtual ~ConnectedLayer();

    void forward();
    void backward();
    void update();

    size_t index;
    size_t size;
    float(*ActivationFunction)(float);
    float* input;
    // input * output
    float* wieght;
    float* output;
};

#endif