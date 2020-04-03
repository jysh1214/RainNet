#include "convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t row, size_t col, size_t filters,
                                       size_t padding, size_t stride,
                                       std::string activation) {
  this->ActivationFunction = getActivationFunction(activation);
  this->ActivationGradient = getActivationGradient(activation);
  this->type = "ConvolutionalLayer";
  this->row = row;
  this->col = col;
  this->filters = filters;
  this->padding = padding;
  this->stride = stride;
}

ConvolutionalLayer::~ConvolutionalLayer() {
  if (this->input)
    delete this->input;
  if (this->weight)
    delete this->weight;
  if (this->bias)
    delete this->bias;
  if (this->output)
    delete this->output;
  if (this->error)
    delete this->error;
}

void ConvolutionalLayer::forward(Net *net) {
  for (size_t i = 0; i < this->filters; i++) {
    tensor *sum1 = new tensor((this->input)->row, (this->input)->col, 1);
    tensor *sum2 = new tensor((this->input)->row, (this->input)->col, 1);
    for (size_t j = 0; j < (this->prev->output)->channel; j++) {
      tensor *temp_a = getChannelMatrix((this->prev)->output, j);
      tensor *temp_w = getChannelMatrix(this->weight, j * (this->filters) + i);
      tensor *conv = convolution(temp_a, temp_w, this->padding, this->stride);
      sum2 = matrixAdd(sum1, conv);
      delete temp_a;
      delete temp_w;
      delete conv;
    }
    assignChannelMatrix(this->input, sum2, i);
    delete sum1;
    delete sum2;
  }
  this->input = matrixAdd(this->input, this->bias);

  for (size_t i = 0;
       i < ((this->output)->row * (this->output)->col * this->filters); i++)
    (this->output)->data[i] = this->ActivationFunction((this->input)->data[i]);

  if (this->next)
    (this->next)->forward(net);
  if (!this->next)
    this->backward(net);
}

void ConvolutionalLayer::backward(Net *net) {
  if (this->prev && this->index > 0) {
    this->update(net);
    (this->prev)->backward(net);
  }
}

void ConvolutionalLayer::update(Net *net) {
  size_t outputRow = (this->output)->row;
  size_t outputCol = (this->output)->col;

  if (!this->next) {
    // show cost
    float cost = net->LossFunction(net->target, this->output);
    std::cout << "cost: " << cost << "\n";
    // count error
    for (size_t i = 0; i < (outputRow * outputCol * this->filters); i++) {
      (this->error)->data[i] =
          this->ActivationFunction((this->output)->data[i]) -
          (net->target)->data[i];
      (this->error)->data[i] *=
          this->ActivationGradient((this->input)->data[i]);
    }
    // update weight
    for (size_t i=0; i<(this->error)->channel; i++){
      for (size_t j=0; j<(this->prev)->filters; j++){
        tensor *temp_a = getChannelMatrix((this->prev)->output, j);
        tensor *temp_e = getChannelMatrix(this->error, i);
        tensor *conv = convolution(temp_a, temp_e, this->padding, this->stride);
        size_t currentChannel = i*(this->prev)->filters + j;
        tensor *weightMatrix = getChannelMatrix(this->weight, currentChannel);
        // tensor* delta = weightMatrix - net->learningRate * conv;
        for (size_t r=0;r<(this->weight)->row; r++){
          for (size_t c=0;c<(this->weight)->col; c++){
            weightMatrix->data[r*(col) + c] -= net->learningRate * conv->data[r*(col) + c];
          }
        }
        assignChannelMatrix(this->weight, weightMatrix, currentChannel);
        delete temp_a;
        delete temp_e;
        delete conv;
        delete weightMatrix;
      }
    }
    // update bias
  } else {
    // count error
    for (size_t i = 0; i < this->filters; i++) {
      tensor *sum1 = new tensor((this->input)->row, (this->input)->col, 1);
      tensor *sum2 = new tensor((this->input)->row, (this->input)->col, 1);
      for (size_t j = 0; j < (this->next)->filters; j++) {
        tensor *temp_a = getChannelMatrix((this->next)->error, j);
        tensor *temp_w = getChannelMatrix((this->next)->weight,
                                          j + i * (this->next)->filters);
        tensor *conv = convolution(temp_a, temp_w, this->padding, this->stride);
        sum2 = matrixAdd(sum1, conv);
        delete temp_a;
        delete temp_w;
        delete conv;
      }
      assignChannelMatrix(this->error, sum2, i);
      delete sum1;
      delete sum2;
    }
    // update weight
    // for (size_t i=0; i<(this->error)->channel; i++){
    //   for (size_t j=0; j<(this->prev)->filters; j++){
    //     tensor *temp_a = getChannelMatrix((this->prev)->output, j);
    //     tensor *temp_e = getChannelMatrix(this->error, i);
    //     tensor *conv = convolution(temp_a, temp_e, this->padding, this->stride);
    //     size_t currentChannel = i*(this->prev)->filters + j;
    //     tensor *weightMatrix = getChannelMatrix(this->weight, currentChannel);
    //     // tensor* delta = weightMatrix - net->learningRate * conv;
    //     for (size_t r=0;r<(this->weight)->row; r++){
    //       for (size_t c=0;c<(this->weight)->col; c++){
    //         weightMatrix->data[r*(col) + c] -= net->learningRate * conv->data[r*(col) + c];
    //       }
    //     }
    //     assignChannelMatrix(this->weight, weightMatrix, currentChannel);
    //     delete temp_a;
    //     delete temp_e;
    //     delete conv;
    //     delete weightMatrix;
    //   }
    // }
    // update bias
  }
}
