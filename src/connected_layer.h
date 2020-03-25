#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "net.h"
#include "layer.h"
#include "activation_fumction.h"
#include "loss_function.h"
#include "tensor.h"
#include "print.h"

#include <assert.h>
#include <iostream>

struct ConnectedLayer: public Layer
{
public:
    ConnectedLayer(size_t size, std::string activation);
    virtual ~ConnectedLayer();

    void forward(Net* net);
    void backward(Net* net);
    void update(Net* net);

    size_t getSize();
    size_t getFilters();   // banned
    size_t getKernelRow(); // banned
    size_t getKernelCol(); // banned

private:
    float(*ActivationFunction)(float);
    float(*ActivationGradient)(float);

    size_t size;
};

#endif
