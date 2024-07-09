CC=mpicc
CFLAGS=-O2 -Wall -mp=gpu -fopenmp -lm
BIN_DIR=bin

default: all

all: create_directory \
	 $(BIN_DIR)/main \
	 $(BIN_DIR)/analytical

create_directory:
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: src/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<
