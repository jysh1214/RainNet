#ifndef DATASET_H
#define DATASET_H

#include "tensor.h"

#include <assert.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

struct Dataset
{
    Dataset(const char *filePath);
    virtual ~Dataset();

    tensor *dataTensor;
};

#endif