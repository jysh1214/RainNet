#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "tensor.h"

#include <assert.h>
#include <cmath>
#include <string>
#include <iostream>

typedef float(*LossFunction)(tensor*, tensor*);

static inline float MSE(tensor* target, tensor* predict)
{
    assert(target && predict);

    float loss = 0.0;
    size_t a = target->row;
    size_t b = target->col;
    size_t c = target->channel;

    for (size_t i=0; i<(a*b*c); ++i){
        loss = (target->data[i] - predict->data[i]) * (target->data[i] - predict->data[i]);
    }

    return loss/(a*b*c);
}

static inline float crossEntropy(tensor* target, tensor* predict)
{
    assert(target && predict);
    
    float loss = 0.0;
    size_t a = target->row;
    size_t b = target->col;
    size_t c = target->channel;

    for (size_t i=0; i<(a*b*c); ++i){
        loss += target->data[i]*logf(predict->data[i]);
    }

    return -loss;
}

static inline float binaryCrossEntropy(tensor* target, tensor* predict)
{
    assert(target && predict);

    float loss = 0.0;
    size_t a = target->row;
    size_t b = target->col;
    size_t c = target->channel;

    for (size_t i=0; i<(a*b*c); ++i){
        loss += -(target->data[i]*logf(predict->data[i]) + (1-target->data[i]))*logf(1 - predict->data[i]);
    }

    return loss;
}

static LossFunction getLossFunction(std::string lossFunction)
{
    if (lossFunction == "MSE"){
        return MSE;
    }
    else if (lossFunction == "crossEntropy"){
        return crossEntropy;
    }
    else if (lossFunction == "binaryCrossEntropy"){
        return binaryCrossEntropy;
    }
    else {
        std::cout << "\ngetLossFunction ERROR: No such loss function.\n" << std::endl;
        return 0;
    }
}

#endif