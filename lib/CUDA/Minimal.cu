#include "Minimal.h"

#ifdef __cplusplus
extern "C" {
#endif

int myCudaFree (void * dev_ptr) {
	cudaError_t err = cudaFree(dev_ptr);
	return (int)err;
}

int myCudaFailed (int err) {
	return (cudaError_t)err != cudaSuccess;
}

const char * myCudaGetErrorString(int err) {
	return cudaGetErrorString((cudaError_t)err);
}

int myCudaMalloc(void ** dev_ptr_ptr, size_t data_len) {
	cudaError_t err= cudaMalloc(dev_ptr_ptr, data_len);
	return (int)err;
}

int myCudaMemcpy(void * dst_ptr, void * src_ptr, size_t length, enum myCudaMemcpyKind mykind) {
	enum cudaMemcpyKind kind;
	switch (mykind) {
		case DeviceToHost: kind = cudaMemcpyDeviceToHost; break;
		case HostToDevice: kind = cudaMemcpyHostToDevice; break;
		case DeviceToDevice: kind = cudaMemcpyDeviceToDevice; break;
	}
	cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, kind);
	return (int)err;
}

void myCudaThreadSynchronize() {
	cudaThreadSynchronize();
}

int myCudaGetLastError() {
	cudaError_t err = cudaGetLastError();
	return (int) err;
}

int myCudaPeekAtLastError() {
	cudaError_t err = cudaPeekAtLastError();
	return (int) err;
}

#ifdef __cplusplus
}
#endif
