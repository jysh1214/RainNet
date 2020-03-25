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

    /**
     * for connected layer:
     * @return: the number of the neurons
    */
    virtual size_t getSize() = 0;
    // for convolutional layer
    virtual size_t getFilters() = 0;
    virtual size_t getKernelRow() = 0;
    virtual size_t getKernelCol() = 0;
};

#endif