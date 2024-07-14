CC=mpicc
NVCC=nvcc
CFLAGS=-O2 -fopenmp -lm
NVCCFLAGS=-Xcompiler "-Wall,-O2,-fopenmp" -lm -lcublas -arch=sm_70
BIN_DIR=bin

default: all

clean:
	@rm -rfv $(BIN_DIR)

all: create_directory \
	 $(BIN_DIR)/main  \
	 $(BIN_DIR)/main-cuda

create_directory:
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: src/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/%: src/%.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<
