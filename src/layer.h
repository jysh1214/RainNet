#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

#include <stdlib.h>
#include <string>

struct Net;

struct Layer
{
    Layer* next;
    Layer* prev;

    /**
     * output of layer_l: Z(l) = W(l)a(l-1) + b(l), and
     * a(l) = activation(Z(l))
     * 
     * output = input * weight + bias
    */
    virtual void forward(Net* net) = 0;
    virtual void backward(Net* net) = 0;
    virtual void update(Net* net) = 0;

    virtual std::string getType() = 0;
    virtual void setIndex(size_t i) = 0;
    virtual size_t getIndex() = 0;

    /**
     * for connected layer:
     * @return: the number of the neurons
    */
    virtual size_t getSize() = 0;
    // for convolutional layer
    virtual size_t getFilters() = 0;
    virtual size_t getKernelRow() = 0;
    virtual size_t getKernelCol() = 0;

    virtual void setInput(tensor* input) = 0;
    virtual tensor* getInput() = 0;
    virtual void setWeight(tensor* weight) = 0;
    virtual tensor* getWeight() = 0;
    virtual void printWeight() = 0;
    virtual void setBias(tensor* bias) = 0;
    virtual tensor* getBias() = 0;
    virtual void printBias() = 0;
    virtual void setOutput(tensor* output) = 0;
    virtual tensor* getOutput() = 0;
    virtual void printOutput() = 0;
    virtual void setError(tensor* error) = 0;
    virtual tensor* getError() = 0;
    virtual void printError() = 0;
};

#endif