#Makefile for 2nd attempt at a CUDA Monte carlo simulator by Dan Liew.

#Set compiler (it must be in your PATH variable in your shell.
NVCC = nvcc

#Set final executable name
EXEC_NAME = casey

#Set the default mode.
debug=0

#decide whether to build debugging or optimised code (higher level means more, i.e. --debug 3 provides more debugging than --debug 2 )
ifeq (${debug},1)
  DEBUG_OR_OPT= --debug --device-debug 3
else
  DEBUG_OR_OPT = --optimize 2
endif

# We now set the compiler options. It is VERY VERY important that you read this as using the right compiler
# options can mean the difference between your program working correctly and NOT working correctly AT ALL!
# For more information you should read the nvcc manual (available from the Nvidia website).
#
# Note -arch specifies the virtual architecture and should be set to only one value.
#
# -code specifies what physical GPU architectures (e.g. sm_20) or virtual architectures to target (e.g. compute_20).
# If -code is set to a virtual architecture then final stage compiling is NOT done which will
# result in a start up delay as the CUDA driver must do JIT compilation when running the built executable.
#
# If -code is set to a real architecture then the final stage of compilation is done and so a binary code for that
# architecture is built and embedded in the executable. The executable does then not need to do JIT compiling upon
# execution as the code for the relevant GPU architecture has already been built speeding up initialisation. Note
# compilation will be SLOWER!
#
# We build PTX to compute_13 (-arch option) because we need double support on the GPU. If you don't have double support
# then nvcc will demote device code to float (single) which will cause HUGE problems as the host code will still think its
# operating on the double data type but the device will think its operating on the float data type.
# We specify to build binaries (-code option) for real GPU architectures sm_13 (Hercules) and sm_20 (Bluecrystal).
# You can remove one of the real GPU architectures that you're not targetting it to speed up compilation 
NVCC_OPTS = -arch=compute_13 -code=sm_13,sm_20 ${DEBUG_OR_OPT} --compiler-options -Wall


#Override implicit rules so we generate dependency files
%.o : %.cu
	#compile object
	${NVCC} --compile -o $@ ${NVCC_OPTS} $< 
	#make dependency file for object
	${NVCC} --generate-dependencies ${NVCC_OPTS} $< > $*.dep


OBJECTS = main.o 


#Default target (does linking of objects)
${EXEC_NAME} : ${OBJECTS}
	${NVCC} ${NVCC_OPTS}  $^ --link -o ${EXEC_NAME}

#include prerequesite files in make file
-include $(OBJECTS:.o=.dep) 

.PHONY : clean
clean : 
	rm *.o *.dep ${EXEC_NAME}

