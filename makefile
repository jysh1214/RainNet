.PHONY: clean dirs

GPP=g++ -std=c++11 -Wall
SRC=src
OBJ=obj
EXAMPLE=example
CXXFLAG=-O3 -fopenmp -Wno-unused-result -Wno-unused-function -Isrc/ -Iinclude/

all: dirs $(EXAMPLE)/conv/conv $(EXAMPLE)/nn/nn $(EXAMPLE)/encoder/encoder

# conv
$(EXAMPLE)/conv/conv: $(EXAMPLE)/conv/conv.cpp\
$(OBJ)/connected_layer.o\
$(OBJ)/convolutional_layer.o\
$(OBJ)/dataset.o\
$(OBJ)/net.o
	$(GPP) $^ -o $@ $(CXXFLAG)

# nn
$(EXAMPLE)/nn/nn: $(EXAMPLE)/nn/nn.cpp\
$(OBJ)/connected_layer.o\
$(OBJ)/convolutional_layer.o\
$(OBJ)/dataset.o\
$(OBJ)/net.o
	$(GPP) $^ -o $@ $(CXXFLAG)

# encoder
$(EXAMPLE)/encoder/encoder: $(EXAMPLE)/encoder/encoder.cpp\
$(OBJ)/connected_layer.o\
$(OBJ)/convolutional_layer.o\
$(OBJ)/dataset.o\
$(OBJ)/net.o
	$(GPP) $^ -o $@ $(CXXFLAG)

$(OBJ)/connected_layer.o: $(SRC)/connected_layer.cpp $(SRC)/connected_layer.h
	$(GPP) -c $< -o $@ $(CXXFLAG)

$(OBJ)/convolutional_layer.o: $(SRC)/convolutional_layer.cpp $(SRC)/convolutional_layer.h
	$(GPP) -c $< -o $@ $(CXXFLAG)

$(OBJ)/dataset.o: $(SRC)/dataset.cpp $(SRC)/dataset.h
	$(GPP) -c $< -o $@ $(CXXFLAG)

$(OBJ)/net.o: $(SRC)/net.cpp $(SRC)/net.h
	$(GPP) -c $< -o $@ $(CXXFLAG)

dirs:
	mkdir -p $(SRC) $(OBJ)

clean:
	rm -rf $(OBJ) $(EXAMPLE)/conv/conv $(EXAMPLE)/nn/nn $(EXAMPLE)/encoder/encoder .vscode

state:
	wc src/*
