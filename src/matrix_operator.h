#ifndef MATRIX_OPERATOR_H
#define MATRIX_OPERATOR_H

#include <stdlib.h>

/**
 * matrixMultiplication - matrix a * matrix b
 * @a: ar*al matrix
 * @b: br*bl matrix
 * 
 * NOTE: al = br
*/
static float* matrixMultiplication(float* a, float* b, size_t ar, size_t al, size_t bl)
{
    float* c = (float*) new float[ar*bl];
    
    #pragma omp parallel for
    for (size_t i=0; i<ar; ++i){
        for (size_t j=0; j<bl; ++j){
            c[j + al*i] = 0;
            for (size_t k=0; k<al; ++k){
                #pragma omp atomic
                c[j + al*i] += a[k + al*i] * b[j + al*k];
            }
        }
    }

    return c;
}

#endif