#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <string>

struct Layer
{
    Layer* next;
    Layer* prev;

    virtual void forward(bool training) = 0;
    virtual void backward() = 0;
    virtual void update() = 0;

    virtual std::string getType() = 0;

    virtual void setIndex(size_t i) = 0;
    virtual size_t getIndex() = 0;
    virtual size_t getSize() = 0;
    virtual void setInput(float* input) = 0;
    virtual float* getInput() = 0;
    virtual void setWieght(float* wieght) = 0;
    virtual float* getWieght() = 0;
    virtual void printWieght() = 0;
    virtual void setOutput(float* output) = 0;
    virtual float* getOutput() = 0;
    virtual void printOutput() = 0;

    virtual void setTarget(float* target) = 0;
};

#endif