#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "matrix_operator.h"

#include <cmath>
#include <string>
#include <iostream>

typedef float(*LossFunction)(tensor*, tensor*);

// L1
static inline float L1(tensor* target, tensor* predict)
{
    float loss = 0.0;
    size_t a = target->row;
    size_t b = target->col;
    size_t c = target->channel;

    for (size_t i=0; i<(a*b*c); ++i){
        loss += abs(target->data[i] - predict->data[i]);
    }

    return loss/(a*b*c);
}

static LossFunction getLossFunction(std::string lossFunction)
{
    if (lossFunction == "L1"){
        return L1;
    }

    std::cout << "\ngetLossFunction ERROR: No such loss function.\n" << std::endl;
    return 0;
}

#endif