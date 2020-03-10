#ifndef PRINT_H
#define PRINT_H

#include <assert.h>
#include <iostream>

/**
 * print - print the data as matrix
 * 
 * @data: pointer to data
 * @row: row of matrix
 * @col: col of matrix
*/
static void print(float* data, size_t row, size_t col)
{
    assert(data && "print ERROR: The data missing.");

    for (size_t i=0; i<row; ++i){
        std::cout << "[";
        for (size_t j=0; j<col; ++j){
            if (data[j + col*i] > 0){
                printf(" %.6f", data[j + col*i]);
            }
            else {
                printf("%.6f", data[j + col*i]);
            }
            
            if (j != col-1) std::cout << " ,";
        }
        std::cout << "]";
        std::cout << std::endl;
    }
}

#endif