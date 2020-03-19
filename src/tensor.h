#ifndef TENSOR_H
#define TENSOR_H

#include "print.h"

#include <assert.h>
#include <stdlib.h>
#include <iostream>

/**
 * NOTE: matrix(row, col) = tensor(row, col, 1)
*/
struct tensor
{
    tensor(size_t row, size_t col, size_t channel)
    {
        assert(row != 0 && col != 0 && channel != 0);
        this->row = row;
        this->col = col;
        this->channel = channel;
        data = (float*) new float[row * col * channel];
        for (size_t i=0; i<(row*col*channel); ++i){
            data[i] = 0.0;
        }
    }
    virtual ~tensor()
    {
        delete [] data;
    }
    
    size_t row;
    size_t col;
    size_t channel;
    float* data;
};

/**
 * flip all matrix in the tensor
*/
static tensor* flip(tensor* a)
{
    assert(a);
    tensor* b = new tensor(a->row, a->col, a->channel);

    for (size_t k=0; k<a->channel; ++k){
        for (size_t i=0; i<a->row; ++i){
            for (size_t j=0; j<a->col; ++j){
                b->data[k*(a->row)*(a->col) + i*(a->col) + j] = 
                a->data[k*(a->row)*(a->col) + (a->row - i)*(a->col) + (a->col - j)];
            }
        }
    }

    return b;
}

/**
 * all elements in tensor a times b
*/
static tensor* dot(tensor*a, float b)
{
    assert(a);
    tensor* c = new tensor(a->row, a->col, a->channel);

    for (size_t k=0; k<a->channel; ++k){
        for (size_t i=0; i<a->row; ++i){
            for (size_t j=0; j<a->col; ++j){
                c->data[k*(a->row)*(a->col) + i*(a->col) + j] = 
                a->data[k*(a->row)*(a->col) + i*(a->col) + j] * b;
            }
        }
    }

    return c;
}

static tensor* tensor2matrix(tensor* a, size_t channel)
{
    assert(a);
    tensor* matrix = new tensor((a->row)*(a->col)*(a->channel/channel), channel, 1);

    size_t t = 0;
    for (size_t k=0; k<a->channel; ++k){
        for (size_t i=0; i<a->row; ++i){
            for (size_t j=0; j<a->col; ++j){
                matrix->data[t + (k*(a->row)*(a->col)+i*(a->col)+j)/matrix->row] = 
                a->data[k*(a->row)*(a->col) + i*(a->col) + j];
                t += channel;
                if (((k*(a->row)*(a->col)+i*(a->col)+j)+1)%matrix->row == 0) t = 0;
                
            }
        }
    }

    return matrix;
}

/**
 * matrixMultiplication - matrix a * matrix b = matrix c
 * @return c
*/
static tensor* matrixMultiplication(tensor* a, tensor* b)
{
    assert((a->col == b->row && a->channel == 1 && b->channel == 1) && "matrixMultiplication ERROR: matrix size not match.");
    tensor* c = new tensor(a->row, b->col, 1);
    
    #pragma omp parallel for
    for (size_t i=0; i<(a->row); ++i){
        for (size_t j=0; j<(b->col); ++j){
            c->data[i*(b->col) + j] = 0;
            for (size_t k=0; k<(a->col); ++k){
                #pragma omp atomic
                c->data[i*(a->col) + j] += a->data[i*(a->col) + k] * b->data[k*(a->col) + j];
            }
        }
    }

    return c;
}

static tensor* paddingZero(tensor* m, size_t padding)
{
    tensor* n = new tensor(m->row+2*padding, m->col+2*padding, m->channel);

    for (size_t k=0; k<m->channel; ++k){
        for (size_t i=0; i<m->row; ++i){
            for (size_t j=0; j<m->col; ++j){
                n->data[k*(n->row)*(n->col) + (i+padding)*(n->col) + j+padding] = m->data[k*(m->row)*(m->col) + i*(m->col) + j];
            }
        }
    }

    return n;
}

/**
 * convolution - output = conv(input, kernel)
 * @return output
 * 
 * NOTE: 
 * output_row = (input_row - kernel_row + 2*padding)/stride + 1
 * output_col = (input_col - kernel_col + 2*padding)/stride + 1
*/
static tensor* convolution(tensor* input, tensor* kernel, size_t padding, size_t stride)
{
    assert(input && kernel);
    assert(!(kernel->channel % input->channel));

    size_t outputRow = (input->row - kernel->row + 2*padding)/stride + 1;
    size_t outputCol = (input->col - kernel->col + 2*padding)/stride + 1;
    size_t outputChannel = (kernel->channel)/(input->channel);

    tensor* output = new tensor(outputRow, outputCol, outputChannel);
    tensor* matrix = paddingZero(input, padding);

    if (kernel->row == 1 && kernel->col == 1){
        outputRow = input->row;
        outputCol = input->col;
        matrix = input;
    }

    #pragma omp parallel for
    for (size_t k=0; k<outputChannel; ++k){
        for (size_t i=0; i<outputRow; ++i){
            for (size_t j=0; j<outputCol; ++j){

                float result = 0.0;
                for (size_t kc=(k*input->channel); kc<(k+1)*input->channel; ++kc){ 
                    for (size_t kr=0; kr<kernel->row; ++kr){
                        for (size_t kl=0; kl<kernel->col; ++kl){
                            #pragma omp atomic
                            result += kernel->data[kc*(kernel->row)*(kernel->col) + kr*(kernel->col) + kl] * 
                            matrix->data[(kc%input->channel)*(kernel->row)*(kernel->col) + (kr+i)*(kernel->col) + (kl+j)];
                            // std::cout<< "kernel: "<<kc<<"; input: "<<kc%input->channel<<std::endl;
                        }
                    }
                }
                output->data[k*(outputRow)*(outputCol) + i*(outputCol) + j] = result;
                result = 0.0;
            }
        }
    }

    if (matrix != input) delete matrix;

    return output;
}

#endif