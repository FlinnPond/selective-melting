GNUPLOT_LIBS := -lboost_iostreams -lboost_system -lboost_filesystem
COMPILE_FLAGS := -arch=sm_86 --relocatable-device-code=false
COMPILECUDA := nvcc

main: main.cu vaporize.cuh
	$(COMPILECUDA) $(COMPILE_FLAGS) -o $@ $< $(GNUPLOT_LIBS)
clean:
	rm -f *.o