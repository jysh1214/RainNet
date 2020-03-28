#ifndef PRINT_H
#define PRINT_H

#include <assert.h>
#include <iostream>

/**
 * print - print the data as tensor
 * @data: pointer to data
 * @row: row of the tensor
 * @col: col of the tensor
 * @channel: channel of the tensor
*/
static void print(float *data, size_t row, size_t col, size_t channel)
{
    assert(data && "print ERROR: The data missing.");

    for (size_t k = 0; k < channel; ++k)
    {
        std::cout << "channel: " << "\n";
        for (size_t i = 0; i < row; ++i)
        {
            std::cout << "[";
            for (size_t j = 0; j < col; ++j)
            {
                data[k * (row) * (col) + i * (col) + j] >= 0 ? printf(" %.6f", data[k * (row) * (col) + i * (col) + j]) : printf("%.6f", data[k * (row) * (col) + i * (col) + j]);
                if (j != col - 1)
                    std::cout << " ,";
            }
            std::cout << "]";
            std::cout << "\n";
        }
    }
}

#endif