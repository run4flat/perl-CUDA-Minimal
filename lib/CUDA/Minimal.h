/* Defines the .cu interface */

#ifdef __cplusplus
extern "C" {
#endif

enum myCudaMemcpyKind {
	DeviceToHost,
	HostToDevice,
	DeviceToDevice,
};

int myCudaFree (void * to_free);
int myCudaFailed (int err);
const char * myCudaGetErrorString(int err);
int myCudaMalloc(void ** dev_ptr, size_t data_len);
int myCudaMemcpy(void * dst_ptr, void * src_ptr, size_t length, enum myCudaMemcpyKind kind);
void myCudaThreadSynchronize();
int myCudaPeekAtLastError();
int myCudaGetLastError();

#ifdef __cplusplus
}
#endif
