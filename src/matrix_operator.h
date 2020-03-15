#ifndef MATRIX_OPERATOR_H
#define MATRIX_OPERATOR_H

#include <assert.h>
#include <stdlib.h>
#include <iostream>
/**
 * Row - major
 * 
 * matrix: 1 x N => matrix[N]
 * for (int i=0; i<N; ++i)
 *     matrix[i];
 * 
 * matrix: N x M => matrix[N * M]
 * for (int i=0; i<N; ++i)
 *     for (int j=0; j<M; ++j)
 *         matrix[i*M + j];
 * 
 * matrix: N x M x C => matrix[N * M * C]
 * for (int i=0; i<N; ++i)
 *     for (int j=0; j<M; ++j)
 *         for (int k=0; k<C; ++k)
 *             matrix[i*M*C + j*C + k];
*/

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
    assert((a->col == b->row) && "matrixMultiplication ERROR: matrix size not match.");
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

#endif