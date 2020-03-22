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

static tensor* paddingZero(tensor* m, size_t padding)
{
    tensor* n = new tensor(m->row+2*padding, m->col+2*padding, m->channel);

    for (size_t k=0; k<m->channel; ++k){
        for (size_t i=0; i<m->row; ++i){
            for (size_t j=0; j<m->col; ++j){
                n->data[k*(n->row)*(n->col) + (i+padding)*(n->col) + j+padding] = 
                m->data[k*(m->row)*(m->col) + i*(m->col) + j];
            }
        }
    }

    return n;
}

/**
 * EX:
 * channel1
 * [[1, 2]] => [1, 5]
 * [[3, 4]]    [2, 6]
 * channel2    [3, 7]
 * [[5, 6]]    [4, 8]
 * [[7, 8]]           
*/
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

/***/
static tensor* tensor2matrix(tensor* a, size_t row, size_t col, size_t padding, size_t stride)
{
    assert(a);
    size_t outputRow = (a->row - row + 2*padding)/stride + 1;
    size_t outputCol = (a->col - col + 2*padding)/stride + 1;
    tensor* b = new tensor((outputRow)*(outputCol), (row)*(col)*(a->channel), 1);

    size_t bSize = (b->row)*(b->col);
    size_t bPosition = 0;

    tensor* m_a = paddingZero(a, padding);

    for (size_t i=0; i<m_a->row; ++i){
        for (size_t j=0; j<m_a->col; ++j){
            
            for (size_t k=0; k<m_a->channel; ++k){
                for (size_t x=0; x<row; ++x){
                    for (size_t y=0; y<col; ++y){
                        // std::cout << k*(m_a->row)*(m_a->col) + (x+i)*(m_a->col) + (y+j) << std::endl;
                        b->data[bPosition] = 
                        m_a->data[k*(m_a->row)*(m_a->col) + (x+i)*(m_a->col) + (y+j)];
                        ++bPosition;
                        if (bPosition == bSize-1) goto DONE;
                    }
                }
            }

        }
    }
    
    DONE:
    return b;
}

/**
 * EX:
 * [1, 5]    channel1
 * [2, 6] => [[1, 2]]
 * [3, 7]    [[3, 4]]
 * [4, 8]    channel2
 *           [[5, 6]]
 *           [[7, 8]]
*/
static tensor* matrix2tensor(tensor* a, size_t row, size_t col)
{
    assert(a && a->channel == 1);
    assert(a->row%(row*col) == 0);
    tensor* b = new tensor(row, col, (a->col)*(a->row/(row*col)));

    size_t channel = a->col;
    size_t t = 0;
    for (size_t k=0; k<b->channel; ++k){
        for (size_t i=0; i<b->row; ++i){
            for (size_t j=0; j<b->col; ++j){
                b->data[k*(b->row)*(b->col) + i*(b->col) + j] = 
                a->data[t + (k*(b->row)*(b->col)+i*(b->col)+j)/a->row];
                t += channel;
                if (((k*(b->row)*(b->col)+i*(b->col)+j)+1)%a->row == 0) t = 0;
            }
        }
    }

    return b;
}

/**
 * matrixMultiplication - matrix a * matrix b = matrix c
 * @return c
*/
static tensor* matrixMultiplication(tensor* a, tensor* b)
{
    assert((a->col == b->row && a->channel == 1 && b->channel == 1) && "matrixMultiplication ERROR: matrix size not match.");
    tensor* c = new tensor(a->row, b->col, 1);
    
    for (size_t i=0; i<(a->row); ++i){
        for (size_t j=0; j<(b->col); ++j){
            c->data[i*(b->col) + j] = 0;
            for (size_t k=0; k<(a->col); ++k){
                c->data[i*(b->col) + j] += a->data[i*(a->col) + k] * b->data[k*(a->col) + j];
            }
        }
    }

    return c;
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

    tensor* output;

    if (kernel->row == 1 && kernel->col == 1){
        output = new tensor(input->row, input->col, input->channel);
        for (size_t k=0; k<input->channel; ++k){
            for (size_t i=0; i<input->row; ++i){
                for (size_t j=0; j<input->col; ++j){
                    input->data[k*(input->row)*(input->col) + i*(input->col) + j] *= kernel->data[0];
                }
            }
        }

        return output;
    }

    tensor* x = tensor2matrix(input, kernel->row, kernel->col, padding, stride);
    tensor* w = tensor2matrix(kernel, outputChannel);
    tensor* y = matrixMultiplication(x, w);
    output = matrix2tensor(y, outputRow, outputCol);
    // print(output->data, output->row, output->col, output->channel);

    // #pragma omp parallel for
    // for (size_t k=0; k<outputChannel; ++k){
    //     for (size_t i=0; i<outputRow; ++i){
    //         for (size_t j=0; j<outputCol; ++j){

    //             float result = 0.0;
    //             for (size_t kc=(k*input->channel); kc<(k+1)*input->channel; ++kc){ 
    //                 for (size_t kr=0; kr<kernel->row; ++kr){
    //                     for (size_t kl=0; kl<kernel->col; ++kl){
    //                         #pragma omp atomic
    //                         result += kernel->data[kc*(kernel->row)*(kernel->col) + kr*(kernel->col) + kl] * 
    //                         matrix->data[(kc%input->channel)*(kernel->row)*(kernel->col) + (kr+i)*(kernel->col) + (kl+j)];
    //                         // std::cout<< "kernel: "<<kc<<"; input: "<<kc%input->channel<<std::endl;
    //                     }
    //                 }
    //             }
    //             output->data[k*(outputRow)*(outputCol) + i*(outputCol) + j] = result;
    //             result = 0.0;
    //         }
    //     }
    // }

    return output;
}

#endif