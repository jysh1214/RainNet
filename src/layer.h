#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

#include <stdlib.h>
#include <string>

struct Net;

/**
 * output of layer_l: Z(l) = W(l)a(l-1) + b(l), and
 * a(l) = activation(Z(l))
 * 
 * output = input * weight + bias
*/
struct Layer
{
    Layer* next;
    Layer* prev;

    virtual void forward(Net* net) = 0;
    virtual void backward(Net* net) = 0;
    virtual void update(Net* net) = 0;

    std::string type;
    size_t index;

    tensor* input;
    tensor* weight;
    tensor* bias;
    tensor* output;
    tensor* error;

    // for connected layer
    size_t size;

    // for convolutional layer
    size_t filters;
    size_t row;
    size_t col;
    size_t padding;
    size_t stride;
};

#endif