#include "dataset.h"

Dataset::Dataset(const char* filePath)
{
    struct stat _st;
    assert(lstat(filePath, &_st) != -1 && "File is not found.");

    std::ifstream cvsFile(filePath);
    std::string data((std::istreambuf_iterator<char>(cvsFile)),
                      std::istreambuf_iterator<char>());

    this->dataTensor = new tensor(800, 800, 1);

    size_t p = 0;
    size_t j = 0;
    for (std::string::size_type i = 0; i < data.size(); ++i){
        if (data[i]==',' || data[i]=='\n' || i==data.size()){
            std::string num_str = data.substr(p, i-p);
            int num = std::stoi(num_str);
            dataTensor->data[j] = num;
            p = i + 1;
            j ++;
        }
    }

    cvsFile.close();
}

Dataset::~Dataset()
{
    delete this->dataTensor;
}