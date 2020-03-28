#ifndef SOFTMAX_H
#define SIFTMAX_H

#include "tensor.h"

#include <cmath>

// softmax: only used in the output layer
static inline tensor *softmax(tensor *x)
{
    size_t a = x->row;
    size_t b = x->col;
    size_t c = x->channel;

    float sum = 0.0;
    for (size_t i = 0; i < (a * b * c); ++i)
    {
        sum += exp(x->data[i]);
    }

    tensor *output = new tensor(a, b, c);
    for (size_t i = 0; i < (a * b * c); ++i)
    {
        output->data[i] = exp(x->data[i]) / sum;
    }

    return output;
}

static inline tensor *softmaxGradient(tensor *x)
{
    size_t a = x->row;
    size_t b = x->col;
    size_t c = x->channel;

    tensor *output = new tensor(a, b, c);
    for (size_t i = 0; i < (a * b * c); ++i)
    {
        output->data[i] = x->data[i];
    }

    size_t maxIndex = 0;
    for (size_t i = 0; i < (a * b * c); ++i)
    {
        if (output->data[i] > output->data[maxIndex])
        {
            maxIndex = i;
        }
    }
    output->data[maxIndex] -= 1;

    return output;
}

#endif