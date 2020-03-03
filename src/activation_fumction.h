#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "math.h"

static inline float sigmoid(float x)
{
    return (exp(x)) / (exp(x) + 1);
}

#endif