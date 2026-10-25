# CS6886 -- Optimising attention on a CPU
#
#   make <variant>   compile src/attn_<variant>.c and run it on every input
#   make all         do that for every variant (missing or broken ones are skipped)
#   make check       compare every variant's output with naive's output
#   make clean       remove build/ and output/
#
# Variants: naive loops tiled fused flash omp_unfused omp_fused
# Compiler: clang by default; override with e.g. `make all CC=gcc`.
# Threads for the OpenMP variants: e.g. `OMP_NUM_THREADS=8 make omp_fused`.

ifeq ($(origin CC),default)
CC = clang
endif
CFLAGS   = -O3 -march=native -std=c11
OMPFLAGS = -fopenmp
LDLIBS   = -lm

VARIANTS = naive loops tiled fused flash omp_unfused omp_fused
INPUTS   = $(wildcard input/*.bin)

.PHONY: all check clean $(VARIANTS)

all: $(VARIANTS)

# Compile one variant, then run it on every input. Outputs go to output/
# (old outputs of the variant are deleted first, so a failed run leaves none).
$(VARIANTS):
	@rm -f build/attn_$@ output/$@_*.bin; \
	if [ ! -f src/attn_$@.c ]; then echo "$@: src/attn_$@.c not found, skipping"; exit 0; fi; \
	mkdir -p build output; \
	flags="$(CFLAGS)"; case $@ in omp_*) flags="$$flags $(OMPFLAGS)";; esac; \
	echo "$(CC) $$flags -o build/attn_$@ src/attn_$@.c $(LDLIBS)"; \
	if ! $(CC) $$flags -o build/attn_$@ src/attn_$@.c $(LDLIBS); then \
	    echo "$@: compile failed, skipping"; exit 0; fi; \
	for f in $(INPUTS); do \
	    ./build/attn_$@ $$f output/$@_`basename $$f` || echo "$@: failed on $$f"; \
	done

# Every variant's output must match naive's: max|O - O_naive| / max|O_naive| <= 1e-4.
check:
	@python3 -c "$$CHECK_PY"

clean:
	rm -rf build output

define CHECK_PY
import os, struct
from array import array

def load(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        a = array("f")
        a.fromfile(f, n * d)
    return a

variants = "$(VARIANTS)".split()
compared = 0
for inp in sorted(os.listdir("input")):
    ref_path = os.path.join("output", "naive_" + inp)
    if not os.path.exists(ref_path):
        print("%-22s no naive output (run make naive first)" % inp)
        continue
    ref = load(ref_path)
    scale = max(abs(x) for x in ref) or 1.0
    for v in variants[1:]:
        path = os.path.join("output", v + "_" + inp)
        if not os.path.exists(path):
            if os.path.exists(os.path.join("src", "attn_" + v + ".c")):
                print("%-22s %-12s NO OUTPUT (did not compile or crashed; run make %s)" % (inp, v, v))
            continue
        out = load(path)
        if len(out) != len(ref):
            print("%-22s %-12s FAIL (wrong size)" % (inp, v))
            continue
        err = max(abs(a - b) for a, b in zip(out, ref)) / scale
        if err != err:
            err = float("inf")
        print("%-22s %-12s %s  (max rel. error %.2e)" % (inp, v, "PASS" if err <= 1e-4 else "FAIL", err))
        compared += 1
if not compared:
    print("nothing to compare yet: run make all (needs naive and at least one other variant)")
endef
export CHECK_PY
