#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "net.h"
#include "layer.h"
#include "activation_fumction.h"
#include "tensor.h"
#include "print.h"

struct ConvolutionalLayer: public Layer
{
public:
    ConvolutionalLayer(size_t row, size_t col, size_t filters, size_t padding, size_t stride, std::string activation);
    virtual ~ConvolutionalLayer();

    void forward(Net* net);
    void backward(Net* net);
    void update(Net* net);

    size_t getSize(); // banned
    size_t getFilters();
    size_t getKernelRow();
    size_t getKernelCol();
    

private:
    float(*ActivationFunction)(float);
    float(*ActivationGradient)(float);

    size_t filters;
    size_t kernelRow;
    size_t kernelCol;
    size_t padding;
    size_t stride;
};

#endif