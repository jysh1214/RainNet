#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "tensor.h"

#include <cmath>
#include <string>
#include <iostream>

typedef float (*ActivationFunction)(float);
typedef float (*ActivationGradient)(float);

static inline float linear(float x)
{
    return x;
}

static inline float linearGradient(float x)
{
    return 1;
}

static inline float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline float sigmoidGradient(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

static inline float relu(float x)
{
    return (x * (x > 0));
}

static inline float reluGradient(float x)
{
    return (x > 0);
}

static inline float leaky(float x)
{
    return (x > 0) ? x : (0.1 * x);
}

static inline float leakyGradient(float x)
{
    return (x > 0) ? 1 : 0.1;
}

static inline float tanh(float x)
{
    return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

static inline float tanhGradient(float x)
{
    return (1 - x * x);
}

static ActivationFunction getActivationFunction(std::string activation)
{
    if (activation == "linear")
    {
        return linear;
    }
    else if (activation == "sigmoid")
    {
        return sigmoid;
    }
    else if (activation == "relu")
    {
        return relu;
    }
    else if (activation == "leaky")
    {
        return leaky;
    }
    else if (activation == "tanh")
    {
        return tanh;
    }
    else if (activation == "INPUT_DATA")
    {
        /* input data */
        return 0;
    }
    else
    {
        std::cout << "\ngetActivationFunction ERROR: No such activation function.\n";
        return 0;
    }
}

static ActivationGradient getActivationGradient(std::string activation)
{
    if (activation == "linear")
    {
        return linearGradient;
    }
    else if (activation == "sigmoid")
    {
        return sigmoidGradient;
    }
    else if (activation == "relu")
    {
        return reluGradient;
    }
    else if (activation == "leaky")
    {
        return leakyGradient;
    }
    else if (activation == "tanh")
    {
        return tanhGradient;
    }
    else if (activation == "INPUT_DATA")
    {
        /* input data */
        return 0;
    }
    else
    {
        std::cout << "\ngetActivationGradient ERROR: No such activation function.\n";
        return 0;
    }
}

#endif