#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"
#include "Minimal.h"

MODULE = CUDA::Minimal		PACKAGE = CUDA::Minimal		

void
_free(SV * dev_ptr_SV)
	CODE:
		// Only free the memory if the pointer is not null:
		if (SvIV(dev_ptr_SV) != 0) {
			// Cast the SV to a device pointer:
			void * dev_ptr = INT2PTR(void*, SvIV(dev_ptr_SV));
			// Free the memory:
			int err = myCudaFree(dev_ptr);
			// Croak on failure.
			if (myCudaFailed(err))
				croak("Unable to free memory on the device: %s"
							, myCudaGetErrorString(err));
			// Set SV to have a value of zero to prevent accidental double frees:
			sv_setiv(dev_ptr_SV, 0);
		}

SV *
_malloc(SV * data_SV)
	CODE:
		// First thing's first: guard against calls in void context:
		if (GIMME_V == G_VOID)
			croak("Cannot call Malloc in void context");
		void * dev_ptr = 0;
		size_t data_len = 0;
		// Check the input arguments:
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
		int err = myCudaMalloc(&dev_ptr, data_len);
		// Check for errors:
		if (myCudaFailed(err))
			croak("Unable to allocate %lu bytes on the device: %s"
						, (long unsigned)data_len, myCudaGetErrorString(err));
		// Set the return:
		RETVAL = newSViv(PTR2IV(dev_ptr));
	OUTPUT:
		RETVAL


void
_transfer(SV * src_SV, SV * dst_SV, ...)
	PROTOTYPE: $$;$$
	CODE:
		void * dst_ptr = 0;
		void * src_ptr = 0;
		size_t length = 0;
		size_t host_offset = 0;
		size_t host_length = 0;
		enum myCudaMemcpyKind kind;
		
	// dump the current contents
	printf("Source SV (at %p):\n", src_SV);
	sv_dump(src_SV);
	printf("Destination SV (at %p):\n", dst_SV);
	sv_dump(dst_SV);
	
		// Get the specified length and host offset if they passed it in:
		if (items > 2) length = (size_t)SvIV(ST(2));
		if (items > 3) host_offset = (size_t)SvIV(ST(3));
		
		// Determine if either of the two SVs are the host memory:
		if (SvPOK(dst_SV) && SvPOK(src_SV)) {
			// We can't have both of them looking like host memory:
			croak("Transfer requires one or more of %s\n%s"
							, "the arguments to be a device pointer"
							, "but it looks like both are host arrays");
		}
		else if (SvPOK(dst_SV)) {
			// Looks like the destination is host memory.
			kind = DeviceToHost;
			host_length = (size_t)SvCUR(dst_SV) - host_offset;
			// set the source (device) and destiantion (host)
			src_ptr = INT2PTR(void*, SvIV(src_SV));
			dst_ptr = SvPVX(dst_SV) + host_offset;
			
			// Get length and make sure the offset is shorter than the host length:
			if (host_length <= 0)
				croak("Host offset must be less than the host's length");
		}
		else if (SvPOK(src_SV)) {
			// Looks like the source is host memory.
			kind = HostToDevice;
			host_length = (size_t)SvCUR(src_SV) - host_offset;
			
			// Set the source (host) and destination (device)
	printf("src address is %p\n", SvPVX(src_SV));
			src_ptr = SvPVX(src_SV);
	if (src_ptr == 0) {printf("Just got null src pointer?\n");
		sv_dump(src_SV);}
			dst_ptr = INT2PTR(void*, SvIV(dst_SV));
			
			// Get length and make sure the offset is shorter than the host length:
		// host_avail_length = host_full_length - host_offset;
			if (host_length <= 0)
				croak("Host offset must be less than the host's length");
		}
		else {
			// Looks like both the source and destination are device pointers.
			kind = DeviceToDevice;
			src_ptr = INT2PTR(void*, SvIV(src_SV));
			dst_ptr = INT2PTR(void*, SvIV(dst_SV));
			if (host_offset > 0) {
				croak("Host offsets are not allowed for %s"
						, "device-to-device transfers");
			}
		}
		
		// Make sure that they provided a length of some sort
		if (length == 0 && host_length == 0)
			croak("You must provide the number of bytes %s"
						, "for device-to-device transfers");
		
		// Make sure the requested length does not exceed the host's length
		if (host_length > 0 && length > host_length)
			croak("Attempting to transfer more data %s"
						, "than the host can accomodate");
		
		// Use the host length if no length was explicitly given:
		if (length == 0) length = host_length;
		
		// Ensure we don't have null pointers
		if (dst_ptr == 0)
			croak("Attempting to transfer to a null pointer");
		if (src_ptr == 0)
			croak("Attempting to transfer from a null pointer");
		
		// Perform the copy and check for errors:
	printf("Copying to destination %p from source at %p\n",
	dst_ptr, src_ptr);
		int err = myCudaMemcpy(dst_ptr, src_ptr, length, kind);
		if (myCudaFailed(err))
			croak("Unable to copy memory: %s"
						, myCudaGetErrorString(err));

void
ThreadSynchronize()
	CODE:
		myCudaThreadSynchronize();

SV *
GetLastError()
	CODE:
		int err = myCudaGetLastError();
		RETVAL = newSVpv(myCudaGetErrorString(err), 0);
	OUTPUT:
		RETVAL

SV *
PeekAtLastError()
	CODE:
		int err = myCudaPeekAtLastError();
		RETVAL = newSVpv(myCudaGetErrorString(err), 0);
	OUTPUT:
		RETVAL
