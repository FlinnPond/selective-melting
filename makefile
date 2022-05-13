GNUPLOT_LIBS := -lboost_iostreams -lboost_system -lboost_filesystem
COMPILE_FLAGS := -arch=sm_50 --relocatable-device-code=true -Wno-deprecated-gpu-targets
COMPILECUDA := nvcc

main: main.cu vaporize.cuh
	$(COMPILECUDA) $(COMPILE_FLAGS) -o $@ $< $(GNUPLOT_LIBS)
# main.o: main.cu vaporize.cuh
# 	$(COMPILECUDA) $(COMPILE_FLAGS) -c -o $@ $<
# vaporize.o: vaporize.cu vaporize.cuh
# 	$(COMPILECUDA) $(COMPILE_FLAGS) -c -o $@ $<
clean:
	rm -f *.o