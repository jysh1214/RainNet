#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "net.h"
#include "layer.h"
#include "activation_fumction.h"
#include "tensor_operator.h"
#include "print.h"

struct ConvolutionalLayer: public Layer
{
public:
    ConvolutionalLayer(size_t filters, size_t row, size_t col, size_t padding, size_t stride, std::string activation);
    virtual ~ConvolutionalLayer();

    void forward(Net* net);
    void backward(Net* net);
    void update(Net* net);

    std::string getType();
    void setIndex(size_t i);
    size_t getIndex();
    size_t getSize(); // banned
    size_t getFilters();
    size_t getKernelRow();
    size_t getKernelCol();
    
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
    size_t filters;
    size_t kernelRow;
    size_t kernelCol;
    size_t padding;
    size_t stride;

    tensor* input;
    tensor* weight;
    tensor* output;
};

#endif