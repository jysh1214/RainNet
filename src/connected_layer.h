#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "layer.h"
#include "activation_fumction.h"
#include "matrix_operator.h"

#include <assert.h>
#include <iostream>

struct ConnectedLayer: public Layer
{
public:
    ConnectedLayer(size_t size, std::string activation);
    virtual ~ConnectedLayer();

    void forward(bool training);
    void backward(float error);
    void update(float error);

    std::string getType();
    void setIndex(size_t i);
    size_t getIndex();
    size_t getSize();
    
    void setInput(float* input);
    float* getInput();
    void setWieght(float* wieght);
    float* getWieght();
    void printWieght();
    void setOutput(float* output);
    float* getOutput();
    void printOutput();

    void setTarget(float* target);

private:
    float(*ActivationFunction)(float);
    float(*ActivationGradient)(float);
    std::string type;
    size_t index;
    size_t size;

    float* input;
    float* wieght;
    float* output;
    float* target;
};

#endif
