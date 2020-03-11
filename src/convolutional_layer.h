#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "net.h"
#include "layer.h"
#include "activation_fumction.h"
#include "matrix_operator.h"
#include "print.h"

struct ConvolutionalLayer: public Layer
{
public:
    ConvolutionalLayer();
    virtual ~ConvolutionalLayer();

    void forward(Net* net);
    void backward(Net* net);
    void update(Net* net);

private:
};

#endif