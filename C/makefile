CC=mpicc
CFLAGS=-O2 -Wall -fopenmp -lm
BIN_DIR=bin

default: all

clean:
	@rm -rfv $(BIN_DIR)

all: create_directory \
	 $(BIN_DIR)/main \
	 $(BIN_DIR)/speedyboi

create_directory:
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/speedyboi: src/speedyboi.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -mavx -o $@ $<

$(BIN_DIR)/%: src/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<
