#include "connected_layer.h"

ConnectedLayer::ConnectedLayer(size_t size, std::string activation): size(size)
{
    ActivationFunction = getActivationFunction(activation);

    /*init wieght randomly
    * range: [-1, 1]
    * size: this_layer * next_layer
    */
    // wieght = (float*) new float[this->size, 1];

}

ConnectedLayer::~ConnectedLayer()
{
    if (input) delete [] input;
    if (wieght) delete [] wieght;
    if (output) delete [] output;
}

void ConnectedLayer::forward()
{

}

void ConnectedLayer::backward()
{

}

void ConnectedLayer::update()
{

}
