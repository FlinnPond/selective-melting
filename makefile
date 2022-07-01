GNUPLOT_LIBS := -lboost_iostreams -lboost_system -lboost_filesystem
COMPILE_FLAGS := -arch=sm_86 --relocatable-device-code=false
COMPILECUDA := nvcc

main: main.cu vaporize.cuh
	$(COMPILECUDA) $(COMPILE_FLAGS) -o $@ $< $(GNUPLOT_LIBS)
# main.o: main.cu vaporize.cuh
# 	$(COMPILECUDA) $(COMPILE_FLAGS) -c -o $@ $<
# vaporize.o: vaporize.cu vaporize.cuh
# 	$(COMPILECUDA) $(COMPILE_FLAGS) -c -o $@ $<
clean:
	rm -f *.o