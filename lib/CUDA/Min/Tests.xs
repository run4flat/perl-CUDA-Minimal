#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"

//////////////////////////////////
// Preprocessor macro constants //
//////////////////////////////////

// These are exported in the BOOT section below.
#define N_THREADS 16
// As designed, each block must be given at least 2 x N_THREADS = 32 floats:
#define MIN_PER_BLOCK 32
// The number of reduction layers in the kernel. Five layers at 16 threads per
// block means a single block can sum over 33 million elements. Allocating 8
// layers will allow me to safely compute the sum of seven+ layers, which would
// involve over 34 gigafloats. There is no reason (at this time of writing) for
// a single threadblock to handle 34 gigafloats of data, so this should be safe:
#define N_LAYERS 8

// This is part of a hack prevent the BOOT section from being included in
// Min.xs's boot section, an error I do not understand at all:
#define TESTS_BOOT

/////////////////
// The kernels //
/////////////////

// Sums all the data in data_g and saves the result in output_g. To keep this
// simple, it assumes length is a multiple of MIN_PER_BLOCK, a fact that must be
// validated by the wrapper function.
__global__ void sum_reduce_kernel(float * data_g, int length, float * output_g) {
	int N_blocks = gridDim.x;			// number of blocks
	int N_to_sum = length / N_blocks;	// number of elements this block will sum
	// The offset in global memory for this particular thread:
	int offset = threadIdx.x + N_to_sum * blockIdx.x;
	
	// The reduction will proceed as follows. For each round, the 16 threads
	// will pull in (2 x N_THREADS) values and store them in a temporary shared
	// working space. The threads will sum those (2 x N_THREADS) values in 5
	// steps and store the result in a list of partial sums. That list itself
	// has room for (2 x N_THREADS) partial sums, and once that list is full,
	// the threads will sum those (2 x N_THREADS) values in 5 steps and store
	// the result in yet another list of partial sums. As you can see, this is a
	// layered approach. The number of layers is N_LAYERS and the shared
	// working memory is this:
	__shared__ float partial_sums[N_LAYERS][2*N_THREADS];
	// Keep track of the current offset for each layer seperately:
	__shared__ int layer_offsets[N_LAYERS];
	
	// **NOTE** The offsets will be incremented each time a new element is added
	// to a given layer, but it's a shared resource so only one thread should do
	// the incrementing at a time.
	
	// Keep track of the current layer:
	int current_layer = 0;
	
	// Before anything else happens, set the shared memory to all zeros:
	for (current_layer = 0; current_layer < N_LAYERS; current_layer++) {
		partial_sums[current_layer][threadIdx.x] = 0.0f;
		partial_sums[current_layer][threadIdx.x + N_THREADS] = 0.0f;
		layer_offsets[current_layer] = 0;
	}
	
	// Let's begin. Iterate as long as there are elements in global memory yet 
	// to be summed:
	current_layer = 0;
	while (N_to_sum > 0) {
		__syncthreads();
		// For the first round of this loop, current_layer == 0, but that need
		// not be the case generally. If it is zero, then load data from global
		// memory into the first layer of partial_sums:
		if (current_layer == 0) {
			partial_sums[0][threadIdx.x] = data_g[offset];
			offset += N_THREADS;
			partial_sums[0][threadIdx.x + N_THREADS] = data_g[offset];
			offset += N_THREADS;
		}
		
		// By construction, at this point the current layer is full, so perform
		// the reduction. (This, by the way, is an unrolled loop, a common idiom
		// in CUDA-C.)
		__syncthreads();
		partial_sums[current_layer][threadIdx.x]
			+= partial_sums[current_layer][threadIdx.x + 16];
		__syncthreads();
		if (threadIdx.x < 8)
			partial_sums[current_layer][threadIdx.x]
				+= partial_sums[current_layer][threadIdx.x + 8];
		__syncthreads();
		if (threadIdx.x < 4)
			partial_sums[current_layer][threadIdx.x]
				+= partial_sums[current_layer][threadIdx.x + 4];
		__syncthreads();
		if (threadIdx.x < 2)
			partial_sums[current_layer][threadIdx.x]
				+= partial_sums[current_layer][threadIdx.x + 2];
		__syncthreads();
		if (threadIdx.x == 0)
			partial_sums[current_layer][0] += partial_sums[current_layer][1];
		__syncthreads();
		
		// I started with 32 elements; now partial_sums[current_layer][0]
		// contains their sum. Store that sum in the next layer. (I could have
		// combined this conditional with the previous one, but I wanted to keep
		// the code for the reduction and the post-reduction bookkeeping
		// separate.)
		if (threadIdx.x == 0) {
			// Store the partial sum in the next layer:
			partial_sums[current_layer + 1][ layer_offsets[current_layer+1] ]
				= partial_sums[current_layer][0];
			// Update the next layer's offset and set this layer's offset to
			// zero:
			layer_offsets[current_layer+1]++;
			layer_offsets[current_layer] = 0;
		}
		__syncthreads();

		// Clear out this layer unless it's the top layer, which is always
		// overwritten and ignored by the final summation:
		if (current_layer > 0) {
			partial_sums[current_layer][threadIdx.x] = 0.0f;
			partial_sums[current_layer][threadIdx.x + N_THREADS] = 0.0f;
		}
		// If this is the top layer, update the number of elements that remain
		// to be summed:
		else {
			N_to_sum -= 2*N_THREADS;
		}

		// If the next layer is full, set it for the next round of summation:
		if (layer_offsets[current_layer+1] == 2*N_THREADS) current_layer++;
		// Otherwise, go back to the top layer:
		else current_layer = 0;
	}
	
	// I have now summed all the data that was passed in. The final step is to
	// accumulate the results by summing up all the values in the various
	// layers. The variable accumulation will hold the sum at each layer, and it
	// will be added to the first element of the next layer. It will also end up
	// with the final sum. (I wonder if I should use a double instead of a float
	// for accumulation?)
	float accumulation = 0.0f;
	for (current_layer = 1; current_layer < N_LAYERS; current_layer++) {
		// Add the previous accumulations to the current layer's zeroeth element
		__syncthreads();
		if (threadIdx.x == 0)
			partial_sums[current_layer][0] += accumulation;
		
		// Sum the values in this layer using the unrolled loop:
		__syncthreads();
		partial_sums[current_layer][threadIdx.x]
			+= partial_sums[current_layer][threadIdx.x + 16];
		__syncthreads();
		if (threadIdx.x < 8)
			partial_sums[current_layer][threadIdx.x]
				+= partial_sums[current_layer][threadIdx.x + 8];
		__syncthreads();
		if (threadIdx.x < 4)
			partial_sums[current_layer][threadIdx.x]
				+= partial_sums[current_layer][threadIdx.x + 4];
		__syncthreads();
		if (threadIdx.x < 2)
			partial_sums[current_layer][threadIdx.x]
				+= partial_sums[current_layer][threadIdx.x + 2];
		__syncthreads();
		// The last step simply stores the results in accumulation
		if (threadIdx.x == 0)
			accumulation = partial_sums[current_layer][0] + partial_sums[current_layer][1];
		__syncthreads();
	}
	
	// Finally, save the results in the provided global memory:
	if (threadIdx.x == 0)
		output_g[blockIdx.x] = accumulation;
}

// A super-simple kernel that just multiplies the values by the supplied
// constant. This modies the data in-place.
__global__ void multiply_by_constant_kernel(float * data_g, float constant) {
	// The offset in global memory for this particular thread:
	int offset = threadIdx.x
				 + blockIdx.x * blockDim.x
				 + blockIdx.y * (gridDim.x * blockDim.x)
				 ;
	data_g[offset] *= constant;
}

// A kernel meant to always succeed no matter how it's invoked.
__global__ void kernel_succeed() {
	float testval = (float)threadIdx.x;
	testval++;
}

/////////////////////////////////
// Host-based utility function //
/////////////////////////////////

float divide_and_conquer_sum (float * data, int length) {
	// Assumes that length is a power of 2, since that is necessary for the
	// kernel anyway:
	if (length == 1) return data[0];
	length /= 2;
	return (divide_and_conquer_sum(data, length)
			+ divide_and_conquer_sum(data + length, length));
}

MODULE = CUDA::Min::Tests		PACKAGE = CUDA::Min::Tests		

########################
# Export the constants #
########################

BOOT:
// This boot seciton does NOT belong in Min.c, but it somehow ends up there
// anyway! This #ifdef prevents stupidity from happening:
#ifdef TESTS_BOOT
	# Add the preprocessor constants to the namespace:
	HV * stash;
	stash = gv_stashpv("CUDA::Min::Tests", TRUE);
	newCONSTSUB(stash, "N_THREADS",	newSViv( N_THREADS ));
	newCONSTSUB(stash, "MIN_PER_BLOCK", newSViv( MIN_PER_BLOCK ));
	newCONSTSUB(stash, "N_LAYERS", newSViv( N_LAYERS ));
#endif

####################################
# Kernel wrapper and Gold function #
####################################

# Reduces the data in dev_data to a collection of sums, stored in dev_output.
# Since inter-block communication is heavily discouraged (except when using
# thread relaunch as a barrier synchronization), this computes one sum for each
# block. To sum all the elements in an array, use N_blocks = 1.
void
_cuda_sum_reduce (SV * dev_data_SV, int length, int N_blocks, SV * dev_intermediate_SV)
	PROTOTYPE: $$$$
	CODE:
		// Convert the SV to float pointer:
		float * starting_data = INT2PTR(float*, SvIV(dev_data_SV));
		float * intermediate_data = INT2PTR(float*, SvIV(dev_intermediate_SV));

		// Note: no single dimension can be larger than 65536, but this was
		// already validated Perl-side. Invoke the kernel:
		sum_reduce_kernel<<<N_blocks, N_THREADS>>>(starting_data, length, intermediate_data);


void
_cuda_multiply_by_constant(SV * dev_data_SV, int length, float constant)
	PROTOTYPE: $$$
	CODE:
		// Convert the SV to a float pointer:
		float * dev_ptr = INT2PTR(float*, SvIV(dev_data_SV));
		// Determine the grid dimensions so that each thread is only responsible
		// for a single element. Perl-side validation must ensure that length is
		// a power of 2:
		int N_threads = 512;
		int N_blocks = length / N_threads;
		int x_dim = N_blocks;
		int y_dim = 1;
		if (N_blocks > 65536) {
			x_dim = 65536;
			y_dim = N_blocks / x_dim;
		}
		dim3 dimGrid(x_dim, y_dim, 1);
		// Launch the kernel:
		multiply_by_constant_kernel<<<dimGrid, N_threads>>>(dev_ptr, constant);


# Performs a fatal kernel invocation:
void
fail_test()
	CODE:
		multiply_by_constant_kernel<<<1, 1010>>>(0, 1);

# Performs a simple kernel that always succeeds:
void
succeed_test()
	CODE:
		kernel_succeed<<<1, 10>>>();

# The all-host function that should give identical, or near identical, results.
float
sum_reduce_gold (SV * data_SV)
	PROTOTYPE: $
	CODE:
		float * data = (float*)SvPVX(data_SV);
		int length = SvCUR(data_SV) / sizeof(float);
		RETVAL = divide_and_conquer_sum(data, length);
	OUTPUT:
		RETVAL

