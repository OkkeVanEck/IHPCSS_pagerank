CC=mpicc
CFLAGS=-O2 -Wall -fopenmp -lm -mavx
BIN_DIR=bin

default: all

all: create_directory \
	 $(BIN_DIR)/main \
	 $(BIN_DIR)/analytical \
	 $(BIN_DIR)/speedyboi

create_directory:
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: src/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<
