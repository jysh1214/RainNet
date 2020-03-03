#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "layer.h"
#include "activation_fumction.h"

class ConnectedLayer: public Layer
{
public:
    ConnectedLayer();
    virtual ~ConnectedLayer();

    void forward();
    void backward();
    void update();

private:
    size_t index;
    float(*ActivationFunction)(float);
    float* input;
    float* wieght;
};

#endif