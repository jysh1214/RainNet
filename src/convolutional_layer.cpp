#include "convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t height, size_t width, size_t channel, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->type = "ConvolutionalLayer";
    this->height = height;
    this->width = width;
    this->channel = channel;
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    if (input) delete [] input;
    if (weight) delete [] weight;
    if (output) delete [] output;
}

void ConvolutionalLayer::forward(Net* net)
{

}

void ConvolutionalLayer::backward(Net* net)
{

}

void ConvolutionalLayer::update(Net* net)
{

}

std::string ConvolutionalLayer::getType()
{
    return this->type;
}

void ConvolutionalLayer::setIndex(size_t i)
{

}

size_t ConvolutionalLayer::getIndex()
{
    return this->index;
}

void ConvolutionalLayer::setInput(float* input)
{

}

float* ConvolutionalLayer::getInput()
{

}

size_t ConvolutionalLayer::getSize()
{
    std::cout << "\nConvolutionalLayer::getSize can't be used.\n" << std::endl;
    exit(0);
}

size_t ConvolutionalLayer::getHeight()
{
    return this->height;
}

size_t ConvolutionalLayer::getWidth()
{
    return this->width;
}

size_t ConvolutionalLayer::getChannel()
{
    return this->channel;
}

void ConvolutionalLayer::setweight(float* weight)
{

}

float* ConvolutionalLayer::getweight()
{

}

void ConvolutionalLayer::printweight()
{

}

void ConvolutionalLayer::setOutput(float* output)
{

}

float* ConvolutionalLayer::getOutput()
{

}

void ConvolutionalLayer::printOutput()
{

}