#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"

MODULE = CUDA::Min		PACKAGE = CUDA::Min		

void
_free(SV * dev_ptr_SV)
	CODE:
		// Only free the memory if the pointer is not null:
		if (SvIV(dev_ptr_SV) != 0) {
			// Cast the SV to a device pointer:
			void * dev_ptr = INT2PTR(void*, SvIV(dev_ptr_SV));
			// Free the memory:
			cudaError_t err = cudaFree(dev_ptr);
			// Croak on failure.
			if (err != cudaSuccess)
				Perl_croak(aTHX_ "Unable to free memory on the device: %s"
							, cudaGetErrorString(err));
			// Set SV to have a value of zero to prevent accidental double frees:
			sv_setiv(dev_ptr_SV, 0);
		}

SV *
Malloc(SV * data_SV)
	CODE:
		// First thing's first: guard against calls in void context:
		if (GIMME_V == G_VOID)
			Perl_croak(aTHX_ "Cannot call Malloc in void context");
		void * dev_ptr = 0;
		size_t data_len = 0;
		if (SvTYPE(data_SV) == SVt_PV) {
			// If the host scalar is a PV, use its length:
			data_len = (size_t)SvCUR(data_SV);
		}
		else {
			// Otherwise interpret the scalar as an integer
			// and use it as the length:
			data_len = (size_t)SvIV(data_SV);
		}
		// Allocate the memory:
		cudaError_t err = cudaMalloc(&dev_ptr, data_len);
		// Check for errors:
		if (err != cudaSuccess)
			Perl_croak(aTHX_ "Unable to allocate %lu bytes on the device: %s"
						, (long unsigned)data_len, cudaGetErrorString(err));
		// Set the return:
		RETVAL = newSViv(PTR2IV(dev_ptr));
	OUTPUT:
		RETVAL


void
Transfer(SV * src_SV, SV * dst_SV, ...)
	PROTOTYPE: $$;$
	CODE:
		void * dst_ptr = 0;
		void * src_ptr = 0;
		size_t length = 0;
		size_t host_length = 0;
		enum cudaMemcpyKind kind;
		
		// Get the specified length, if they passed it in:
		if (items == 3) length = (size_t)SvIV(ST(2));
		
		// Determine if either of the two SVs are the host memory:
		if (SvTYPE(dst_SV) == SVt_PV && SvTYPE(src_SV) == SVt_PV) {
			// We can't have both of them looking like host memory:
			Perl_croak(aTHX_ "Transfer requires at least one of %s\n%s"
							, "the arguments to be a device pointer"
							, "but it looks like both are host arrays");
		}
		else if (SvTYPE(dst_SV) == SVt_PV) {
			// Looks like the destination is host memory.
			kind = cudaMemcpyDeviceToHost;
			host_length = (size_t)SvCUR(dst_SV);
			src_ptr = INT2PTR(void*, SvIV(src_SV));
			dst_ptr = SvPVX(dst_SV);
		}
		else if (SvTYPE(src_SV) == SVt_PV) {
			// Looks like the source is host memory.
			kind = cudaMemcpyHostToDevice;
			host_length = (size_t)SvCUR(src_SV);
			src_ptr = SvPVX(src_SV);
			dst_ptr = INT2PTR(void*, SvIV(dst_SV));
		}
		else {
			// Looks like both the source and destination are device pointers.
			kind = cudaMemcpyDeviceToDevice;
			src_ptr = INT2PTR(void*, SvIV(src_SV));
			dst_ptr = INT2PTR(void*, SvIV(dst_SV));
		}
		
		// Make sure that they provided a length of some sort
		if (length == 0 && host_length == 0)
			Perl_croak(aTHX_ "You must provide the number of bytes %s"
						, "for device-to-device transfers");
		
		// Make sure the requested length does not exceed the host's length
		if (host_length > 0 && length > host_length)
			Perl_croak(aTHX_ "Attempting to transfer more data %s"
						, "than the host can accomodate");
		
		// Use the host length if no length was explicitly given:
		if (length == 0) length = host_length;
		
		// Perform the copy and check for errors:
		cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, kind);
		if (err != cudaSuccess)
			Perl_croak(aTHX_ "Unable to copy memory: %s"
						, cudaGetErrorString(err));

void
ThreadSynchronize()
	CODE:
		cudaThreadSynchronize();

SV *
GetLastError()
	CODE:
		cudaError_t err = cudaGetLastError();
		RETVAL = newSVpv(cudaGetErrorString(err), 0);
	OUTPUT:
		RETVAL

