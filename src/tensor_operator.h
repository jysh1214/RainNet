#ifndef MATRIX_OPERATOR_H
#define MATRIX_OPERATOR_H

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