#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "tensor_operator.h"

#include <cmath>
#include <string>
#include <iostream>

typedef float(*LossFunction)(tensor*, tensor*);

// L1: sum(abs(target - predict)) / n
static inline float L1(tensor* target, tensor* predict)
{
    float loss = 0.0;
    size_t a = target->row;
    size_t b = target->col;
    size_t c = target->channel;

    for (size_t i=0; i<(a*b*c); ++i){
        loss += fabs(target->data[i] - predict->data[i]);
    }

    return loss/(a*b*c);
}

// L2: sum((target - predict)^2) / 2n
static inline float L2(tensor* target, tensor* predict)
{
    float loss = 0.0;
    size_t a = target->row;
    size_t b = target->col;
    size_t c = target->channel;

    for (size_t i=0; i<(a*b*c); ++i){
        loss += ((target->data[i]-predict->data[i]) * (target->data[i]-predict->data[i]));
    }

    return loss/(2 * (a*b*c));
}

// cross entropy: sum(target*log(predict) + (1-target)*log(1-predict))/(-n)
static inline float crossEntropy(tensor* target, tensor* predict)
{
    float loss = 0.0;
    size_t a = target->row;
    size_t b = target->col;
    size_t c = target->channel;

    for (size_t i=0; i<(a*b*c); ++i){
        loss += (target->data[i]*logf(predict->data[i]) + (1-target->data[i])*logf(1-predict->data[i]));
    }

    return -loss/(a*b*c);
}

static LossFunction getLossFunction(std::string lossFunction)
{
    if (lossFunction == "L1"){
        return L1;
    }
    else if (lossFunction == "L2"){
        return L2;
    }
    else if (lossFunction == "crossEntropy"){
        return crossEntropy;
    }
    else {
        std::cout << "\ngetLossFunction ERROR: No such loss function.\n" << std::endl;
        return 0;
    }
}

#endif