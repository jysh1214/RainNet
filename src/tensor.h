#ifndef TENSOR_H
#define TENSOR_H

#include "gemm.h"
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
        data = (float *)new float[row * col * channel];
        for (size_t i = 0; i < (row * col * channel); ++i)
        {
            data[i] = 0.0;
        }
    }
    virtual ~tensor()
    {
        delete[] data;
    }

    size_t row;
    size_t col;
    size_t channel;
    float *data;
};

static tensor *matrixAdd(tensor *a, tensor *b)
{
    assert(a->row == b->row);
    assert(a->col == b->col);
    assert(a->channel == b->channel);

    tensor *c = new tensor(a->row, a->col, a->channel);

    for (size_t k = 0; k < a->channel; ++k)
    {
        for (size_t i = 0; i < a->row; ++i)
        {
            for (size_t j = 0; j < a->col; ++j)
            {
                c->data[k * (a->row) * (a->col) + i * (a->col) + j] =
                    a->data[k * (a->row) * (a->col) + i * (a->col) + j] + b->data[k * (a->row) * (a->col) + i * (a->col) + j];
            }
        }
    }

    return c;
}

static tensor *paddingZero(tensor *m, size_t padding)
{
    tensor *n = new tensor(m->row + 2 * padding, m->col + 2 * padding, m->channel);

    for (size_t k = 0; k < m->channel; ++k)
    {
        for (size_t i = 0; i < m->row; ++i)
        {
            for (size_t j = 0; j < m->col; ++j)
            {
                n->data[k * (n->row) * (n->col) + (i + padding) * (n->col) + j + padding] =
                    m->data[k * (m->row) * (m->col) + i * (m->col) + j];
            }
        }
    }

    return n;
}

/**
 * weight tensor to weight matrix
*/
static tensor *tensor2matrix(tensor *a, size_t channel)
{
    assert(a);
    tensor *matrix = new tensor((a->row) * (a->col) * (a->channel / channel), channel, 1);

    size_t t = 0;
    for (size_t k = 0; k < a->channel; ++k)
    {
        for (size_t i = 0; i < a->row; ++i)
        {
            for (size_t j = 0; j < a->col; ++j)
            {
                matrix->data[t + (k * (a->row) * (a->col) + i * (a->col) + j) / matrix->row] =
                    a->data[k * (a->row) * (a->col) + i * (a->col) + j];
                t += channel;
                if (((k * (a->row) * (a->col) + i * (a->col) + j) + 1) % matrix->row == 0)
                    t = 0;
            }
        }
    }

    return matrix;
}

/**
 * prev output tensor to matrix
*/
static tensor *tensor2matrix(tensor *a, size_t row, size_t col, size_t padding, size_t stride)
{
    assert(a);
    size_t outputRow = (a->row - row + 2 * padding) / stride + 1;
    size_t outputCol = (a->col - col + 2 * padding) / stride + 1;
    tensor *b = new tensor((outputRow) * (outputCol), (row) * (col) * (a->channel), 1);

    size_t bSize = (b->row) * (b->col);
    size_t bPosition = 0;

    tensor *m_a = paddingZero(a, padding);

    for (size_t i = 0; i < m_a->row; i += padding)
    {
        for (size_t j = 0; j < m_a->col; j += padding)
        {
            for (size_t k = 0; k < m_a->channel; ++k)
            {
                for (size_t x = 0; x < row; ++x)
                {
                    for (size_t y = 0; y < col; ++y)
                    {
                        // std::cout << k*(m_a->row)*(m_a->col) + (x+i)*(m_a->col) + (y+j) << std::endl;
                        b->data[bPosition] =
                            m_a->data[k * (m_a->row) * (m_a->col) + (x + i) * (m_a->col) + (y + j)];
                        ++bPosition;
                        if (bPosition == bSize - 1)
                            goto DONE;
                    }
                }
            }
        }
    }

DONE:
    return b;
}

/**
* input matrix to input tensor
*/
static tensor *matrix2tensor(tensor *a, size_t row, size_t col)
{
    assert(a && a->channel == 1);
    assert(a->row % (row * col) == 0);
    tensor *b = new tensor(row, col, (a->col) * (a->row / (row * col)));

    size_t channel = a->col;
    size_t t = 0;
    for (size_t k = 0; k < b->channel; ++k)
    {
        for (size_t i = 0; i < b->row; ++i)
        {
            for (size_t j = 0; j < b->col; ++j)
            {
                b->data[k * (b->row) * (b->col) + i * (b->col) + j] =
                    a->data[t + (k * (b->row) * (b->col) + i * (b->col) + j) / a->row];
                t += channel;
                if (((k * (b->row) * (b->col) + i * (b->col) + j) + 1) % a->row == 0)
                    t = 0;
            }
        }
    }

    return b;
}

/**
 * NOTE: i start from 0
*/
static tensor *getChannelMatrix(tensor *a, size_t i)
{
    assert(i <= a->channel);
    tensor *output = new tensor(a->row, a->col, 1);
    for (size_t r = 0; r < a->row; r++)
    {
        for (size_t c = 0; c < a->col; c++)
        {
            output->data[r * (a->col) + c] = a->data[i * (a->row * a->col) + r * (a->col) + c];
        }
    }

    return output;
}

/**
 * assign tensor a channel i = matrix
*/
static void assignChannelMatrix(tensor *a, tensor *matrix, size_t i)
{
    assert(a);
    assert(matrix);
    assert(i <= a->channel);
    assert(matrix->channel == 1);
    assert(a->row == matrix->row && a->col == matrix->col);

    for (size_t r = 0; r < (a->row); r++)
    {
        for (size_t c = 0; c < (a->col); c++)
        {
            a->data[i * (a->row * a->col) + r * (a->col) + c] = matrix->data[r * (a->col) + c];
        }
    }
}

/**
 * matrixMultiplication - matrix a * matrix b = matrix c
 * @a: matrix
 * @TA: transpose matrix a
 * @b: matrix
 * @TB: transpose matrix b
*/
static tensor *matrixMul(tensor *a, int TA, tensor *b, int TB)
{
    assert(a->channel == 1 && b->channel == 1);
    tensor *c = new tensor(a->row, b->col, 1);

    gemm(TA, TB, a->row, b->col, a->col, 1, a->data, a->col, b->data, b->col, 1, c->data, c->col);

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
static tensor *convolution(tensor *input, tensor *kernel, size_t padding, size_t stride)
{
    assert(input && kernel);
    assert(input->channel == 1 && kernel->channel == 1);

    size_t outputRow = (input->row - kernel->row + 2 * padding) / stride + 1;
    size_t outputCol = (input->col - kernel->col + 2 * padding) / stride + 1;
    // size_t outputChannel = (kernel->channel)/(input->channel);

    tensor *output;

    if (kernel->row == 1 && kernel->col == 1)
    {
        output = new tensor(input->row, input->col, input->channel);
        for (size_t k = 0; k < input->channel; ++k)
        {
            for (size_t i = 0; i < input->row; ++i)
            {
                for (size_t j = 0; j < input->col; ++j)
                {
                    input->data[k * (input->row) * (input->col) + i * (input->col) + j] *= kernel->data[0];
                }
            }
        }

        return output;
    }

    tensor *prevOuput = tensor2matrix(input, kernel->row, kernel->col, padding, stride);
    tensor *w = tensor2matrix(kernel, 1);
    tensor *thisInput = matrixMul(prevOuput, 0, w, 0);
    output = matrix2tensor(thisInput, outputRow, outputCol);

    delete prevOuput;
    delete w;
    delete thisInput;

    return output;
}

#endif
