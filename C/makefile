CC=mpicc
CFLAGS=-O2 -Wall -fopenmp -lm
BIN_DIR=bin

default: all

clean:
	@rm -rfv $(BIN_DIR)

all: create_directory \
	 $(BIN_DIR)/main 

create_directory:
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: src/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<
