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

    std::string getType();
    void setIndex(size_t i);
    size_t getIndex();
    size_t getSize();
    size_t getFilters();   // banned
    size_t getKernelRow(); // banned
    size_t getKernelCol(); // banned
    
    void setInput(tensor* input);
    tensor* getInput();
    void setWeight(tensor* weight);
    tensor* getWeight();
    void printWeight();
    void setOutput(tensor* output);
    tensor* getOutput();
    void printOutput();

private:
    float(*ActivationFunction)(float);
    float(*ActivationGradient)(float);
    std::string type;
    size_t index;
    size_t size;

    tensor* input;
    tensor* weight;
    tensor* output;
};

#endif
