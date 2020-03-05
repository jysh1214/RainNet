#include "connected_layer.h"

ConnectedLayer::ConnectedLayer(size_t size, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->type = "ConnectedLayer";
    this->size = size;
}

ConnectedLayer::~ConnectedLayer()
{
    if (input) delete [] input;
    if (wieght) delete [] wieght;
    if (output) delete [] output;
}

void ConnectedLayer::forward()
{
    std::cout << "fuck" << std::endl;
}

void ConnectedLayer::backward()
{

}

void ConnectedLayer::update()
{

}

std::string ConnectedLayer::getType()
{
    return this->type;
}

void ConnectedLayer::setIndex(size_t i)
{
    this->index = i;
}


size_t ConnectedLayer::getIndex()
{
    return this->index;
}

size_t ConnectedLayer::getSize()
{
    return this->size;
}

void ConnectedLayer::setInput(float* input)
{
    this->input = input;
}

float* ConnectedLayer::getInput()
{
    return this->input;
}

void ConnectedLayer::setWieght(float* wieght)
{
    this->wieght = wieght;
}

float* ConnectedLayer::getWieght()
{
    return this->wieght;
}

void ConnectedLayer::setOutput(float* output)
{
    this->output = output;
}

float* ConnectedLayer::getOutput()
{
    return this->output;
}
