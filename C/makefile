CC=mpicc
CFLAGS=-O2 -Wall -fopenmp -lm -mavx
BIN_DIR=bin

default: all

clean:
	@rm -rfv $(BIN_DIR)

all: create_directory \
	 $(BIN_DIR)/main \
	 $(BIN_DIR)/analytical \
	 $(BIN_DIR)/speedyboi \
	 $(BIN_DIR)/speedyboi512 \
	 $(BIN_DIR)/speedyboi-portable

create_directory:
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/speedyboi512: src/speedyboi.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -mavx512f -DAVX512 -o $@ $<

$(BIN_DIR)/%: src/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<
