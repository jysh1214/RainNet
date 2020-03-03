#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>
#include <string>

typedef float(*ActivationFunction)(float);

static inline float sigmoid(float x)
{
    return (exp(x)) / (exp(x) + 1);
}

static ActivationFunction getActivationFunction(std::string activation)
{
    if (activation == "sigmoid"){
        return sigmoid;
    }
}

#endif