#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "layer.h"
#include "activation_fumction.h"

#include <iostream>

struct ConnectedLayer: public Layer
{
public:
    ConnectedLayer(size_t size, std::string activation);
    virtual ~ConnectedLayer();

    struct Layer* next;
    struct Layer* prev;

    void forward();
    void backward();
    void update();

    std::string getType();
    void setIndex(size_t i);
    size_t getIndex();
    size_t getSize();
    
    void setInput(float* input);
    float* getInput();
    void setWieght(float* wieght);
    float* getWieght();
    void setOutput(float* output);
    float* getOutput();

private:
    float(*ActivationFunction)(float);
    std::string type;
    size_t index;
    size_t size;

    float* input;
    float* wieght;
    float* output;
};

#endif