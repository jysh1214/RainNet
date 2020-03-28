#include "convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t row, size_t col, size_t filters, size_t padding, size_t stride, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->row = row;
    this->col = col;
    this->type = "ConvolutionalLayer";
    this->filters = filters;
    this->padding = padding;
    this->stride = stride;
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    if (this->input)
        delete this->input;
    if (this->weight)
        delete this->weight;
    if (this->bias)
        delete this->bias;
    if (this->output)
        delete this->output;
    if (this->error)
        delete this->error;
}

void ConvolutionalLayer::forward(Net *net)
{
}

void ConvolutionalLayer::backward(Net *net)
{
    if (this->prev)
    {
        this->update(net);
        (this->prev)->backward(net);
    }
}

void ConvolutionalLayer::update(Net *net)
{
}
