use Test::More tests => 26;

# This file starts with z_ to ensure that it runs last.

# Load CUDA::Minimal::Tests, which provides functions to facilitate testing
# kernel invocations:
use CUDA::Minimal;
use CUDA::Minimal::Tests;

use strict;
use warnings;

# First make sure that non-failures are correctly handled:
my $string = GetLastError;
is($string, 'no error', "With no error, GetLastError should return 'no error'");
ok(not (ThereAreCudaErrors), 'ThereAreCudaErrors should return false when there are none');
# Make sure the kernel that's supposed to always succeed does, in fact, succeed:
CUDA::Minimal::Tests::succeed_test();
ok(not (ThereAreCudaErrors), 'succeed_test does not set a CUDA error');

# Create a collection of values to sum and copy them to the device:
my $N_elements = 1024;
my $host_array = pack ('f*', 1..$N_elements);
my $dev_ptr = MallocFrom($host_array);

# Run the multiply kernel; copy back 10 random values and make sure they are
# what they are supposed to be:
CUDA::Minimal::Tests::cuda_multiply_by_constant($dev_ptr, $N_elements, 4);
my $test_val = pack 'f', 1;
for (1..10) {
	my $offset = int rand $N_elements;
	Transfer(Sizeof(f=>$offset) + $dev_ptr => $test_val);
	ok(unpack('f', $test_val) == ($offset+1)*4, "Position $offset has value " . ($offset+1)*4);
}

# Return the values to their original state:
CUDA::Minimal::Tests::cuda_multiply_by_constant($dev_ptr, $N_elements, 0.25);

# Test the sum:
CUDA::Minimal::Tests::sum_reduce_test($host_array, $dev_ptr, 'a single block');
CUDA::Minimal::Tests::sum_reduce_test($host_array, $dev_ptr, '32 blocks', 32);

# Free the current device memory before moving forward:
Free($dev_ptr);

# Now I'm going to try a nontrivial sum of 65536 floats:
$host_array = pack 'f*', map {sin ($_/10)} 1..65536;
$dev_ptr = MallocFrom($host_array);
CUDA::Minimal::Tests::sum_reduce_test($host_array, $dev_ptr, 'single block sine-sum');
CUDA::Minimal::Tests::sum_reduce_test($host_array, $dev_ptr, '32 block sine-sum', 32);
CUDA::Minimal::Tests::sum_reduce_test($host_array, $dev_ptr, '1024,32 block sine-sum', 1024, 32);





# Finish by running the failure test. This, among other things, is supposed to
# ensure that the documentation regarding the unspecified launch failure (for
# kernel invocations after a failed kernel launch) is correct. If all the tests
# after the first one fails, the documentation should be updated accordingly.

# Run the kernel that is supposed to fail and see what we get:
CUDA::Minimal::Tests::fail_test();
ThreadSynchronize();
ok(ThereAreCudaErrors, "ThereAreCudaErrors returns a true value when an error occurs");

# ThereAreCudaErrors should return true until GetLastError,
ok(ThereAreCudaErrors, "ThereAreCudaErrors does not clear the last error");

# Should be able to peek at the error:
like(PeekAtLastError, qr/unspecified/, 'PeekAtLastError correctly returns an unspecified launch failure');

# ThereAreCudaErrors should return true until GetLastError,
ok(ThereAreCudaErrors, "PeekAtLastError does not clear the last error");

# Double-check the output of the error (and clear it):
like(GetLastError, qr/unspecified/, "The failing kernel gives an unspecified launch failure");

# further calls should return false (no errors) until I run another kernel:
ok(not (ThereAreCudaErrors), "GetLastError clears the last error");

# Check that the next kernel invocation trips an error
CUDA::Minimal::Tests::succeed_test();
ok(ThereAreCudaErrors, "Good kernels invoked after a failed kernel launch also fail");

# Check the return value of GetLastError:
CUDA::Minimal::Tests::succeed_test();
like(GetLastError, qr/unspecified/, 'Further kernel invocations return an unspecified launch failure');
