use Test::More;

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
ok(not (ThereAreCudaErrors), 'succeed_test does not set a CUDA error')
	or diag('succeed_test() failed with: ' . GetLastError);

# Create a collection of values to sum and copy them to the device:
my $N_elements = 1024;
my $host_array = pack ('f*', 1..$N_elements);
my $dev_ptr = MallocFrom($host_array);
my $test_val = pack 'f', 1;

# Run the multiply kernel; copy back 10 random values and make sure they are
# what they are supposed to be:
CUDA::Minimal::Tests::cuda_multiply_by_constant($dev_ptr, $N_elements, 4);
ok(not (ThereAreCudaErrors), 'cuda_multiply_by_constant does not cause a CUDA error')
	or diag('cuda_multiply_by_constant() failed with: ' . GetLastError);
subtest 'Multiply by four' => sub {
	plan tests => 10;
	for (1..10) {
		my $offset_idx = int rand $N_elements;
		my $offset_byte = Sizeof(f=>$offset_idx);
		Transfer($dev_ptr + $offset_byte => $test_val);
		my $got = unpack('f', $test_val);
		my $expected = ($offset_idx+1)*4;
		is($got, $expected, "Position $offset_idx looks right");
	}
};

# Return the values to their original state:
CUDA::Minimal::Tests::cuda_multiply_by_constant($dev_ptr, $N_elements, 0.25);
ok(not (ThereAreCudaErrors), 'cuda_multiply_by_constant does not cause a CUDA error')
	or diag('cuda_multiply_by_constant() failed with: ' . GetLastError);
subtest 'Restore by multiplying by 0.25' => sub {
	plan tests => 10;
	for (1..10) {
		my $offset_idx = int rand $N_elements;
		my $offset_byte = Sizeof(f=>$offset_idx);
		Transfer($dev_ptr + $offset_byte => $test_val);
		my $got = unpack('f', $test_val);
		my $expected = $offset_idx+1;
		is($got, $expected, "Position $offset_idx looks right");
	}
};

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





# Re-create a collection of values that we will use for kernel invocation
# testing after a failed launch
$host_array = pack ('f*', 1..$N_elements);
$dev_ptr = MallocFrom($host_array);

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

# Cannot allocate new memory
my $malloc_succeeds = eval {
	Free(Malloc(4));
	fail('Malloc fails after a failed kernel launch');
	1;
} or do {
	pass('Malloc fails after a failed kernel launch');
	ok(ThereAreCudaErrors, 'cuda reports errors after said malloc');
	like(GetLastError, qr/unspecified/, 'cuda reports an unspecified launch failure');
};

# Finally, can we invoke a kernel afterwards?
CUDA::Minimal::Tests::succeed_test();
ok(not(ThereAreCudaErrors), 'succeed_test **succeeds** after a failed kernel?!?!');

# Run the multiply kernel; copy back 10 random values and make sure they are
# what they are supposed to be:
CUDA::Minimal::Tests::cuda_multiply_by_constant($dev_ptr, $N_elements, 4);
ok(ThereAreCudaErrors, 'cuda_multiply_by_constant *does* cause a CUDA error *after* thread launch failure')
	or diag('cuda_multiply_by_constant() failed with: ' . GetLastError);

done_testing;
