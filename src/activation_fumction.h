#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>
#include <string>
#include <iostream>

typedef float(*ActivationFunction)(float);
typedef float(*ActivationGradient)(float);

// linear
static inline float linear(float x)
{
    return x;
}

static inline float linearGradient(float x)
{
    return 1;
}

// sigmoid
static inline float sigmoid(float x)
{
    return (exp(x)) / (exp(x) + 1);
}

static inline float sigmoidGradient(float x)
{ 
    return (x * (1 - x));
}

// ReLu
static inline float relu(float x)
{
    return (x * (x > 0));
}

static inline float reluGradient(float x)
{
    return (x > 0);
}

// tanh
static inline float tanh(float x)
{
    return (exp(2*x) - 1)/(exp(2*x) + 1);
}

static inline float tanhGradient(float x)
{
    return (1 - x*x);
}

static ActivationFunction getActivationFunction(std::string activation)
{
    if (activation == "linear"){
        return linear;
    }
    else if (activation == "sigmoid"){
        return sigmoid;
    }
    else if (activation == "relu"){
        return relu;
    }
    else if (activation == "tanh"){
        return tanh;
    }
    
    std::cout << "\ngetActivationFunction ERROR: No such activation function.\n" << std::endl;
    return 0;
}

static ActivationGradient getActivationGradient(std::string activation)
{
    if (activation == "linear"){
        return linearGradient;
    }
    else if (activation == "sigmoid"){
        return sigmoidGradient;
    }
    else if (activation == "relu"){
        return reluGradient;
    }
    else if (activation == "tanh"){
        return tanhGradient;
    }

    std::cout << "\ngetActivationGradient ERROR: No such activation function.\n" << std::endl;
    return 0;
}

#endif