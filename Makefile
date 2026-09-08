# EdgeCore - build for the C inference runtime and hardware profiler.
CC       ?= gcc
CFLAGS   ?= -O3 -march=native -std=c11 -Wall -Wextra -fopenmp
LDFLAGS   = -fopenmp -lm -pthread

BUILD    := build
RUNTIME  := $(BUILD)/edgecore-runtime
HWPROF   := $(BUILD)/edgecore-hwprof

SRCS := src/runtime_main.c src/gpt2.c src/kernels.c src/tokenizer.c
OBJS := $(SRCS:src/%.c=$(BUILD)/%.o)

all: $(RUNTIME) $(HWPROF)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c src/edgecore.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(HWPROF): $(BUILD)/hw_profiler.o
	$(CC) $< -o $@ $(LDFLAGS)

smoke: all
	bash scripts/smoke_test.sh

e2e: all
	python3 -m edgecore.cli e2e --fetch

clean:
	rm -rf $(BUILD)

.PHONY: all smoke e2e clean
