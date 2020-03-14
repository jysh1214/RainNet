#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <string>

struct Net;

struct Layer
{
    Layer* next;
    Layer* prev;

    virtual void forward(Net* net) = 0;
    virtual void backward(Net* net) = 0;
    virtual void update(Net* net) = 0;

    virtual std::string getType() = 0;

    virtual void setIndex(size_t i) = 0;
    virtual size_t getIndex() = 0;

    // connected layer
    virtual size_t getSize() = 0;
    // convolutional layer
    virtual size_t getHeight() = 0;
    virtual size_t getWidth() = 0;
    virtual size_t getChannel() = 0;

    virtual void setInput(float* input) = 0;
    virtual float* getInput() = 0;
    virtual void setWeight(float* weight) = 0;
    virtual float* getWeight() = 0;
    virtual void printWeight() = 0;
    virtual void setOutput(float* output) = 0;
    virtual float* getOutput() = 0;
    virtual void printOutput() = 0;
};

#endif