#ifndef CUDA_FUNCTIONS_CUH
#define CUDA_FUNCTIONS_CUH

typedef unsigned int uint;
typedef unsigned long ulong;

extern "C"
{
    void cudaInit(int argc, const char **argv);
    void cudaDestroy();

	void allocateArray(void **devPtr, int size);
	void freeArray(void *devPtr);

	void copyArrayToDevice(void *device, const void *host, int offset, int size);
	void copyArrayFromDevice(void *host, const void *device, int size);

	ulong sumNumbers(ulong *dNumbers, ulong n);
}

#endif // CUDA_FUNCTIONS_CUH
