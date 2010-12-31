use Test::More tests => 27;

# Load CUDA::Min::Tests, which provides functions to facilitate testing:
use CUDA::Min ':all';
use CUDA::Min::Tests;

die "Working here";

use strict;
use warnings;

# Create a collection of values to sum:
my $N_elements = 1024;
my $host_array = pack ('f*', 1..$N_elements);

# Copy those values to the device:
my $dev_ptr = MallocFrom($host_array);
ok(1, "MallocFrom does not croak");

# Copy back 10 random values and make sure they are what they're supposed to be
my $test_val = pack 'f', 1;
my $sizeof_float = length $test_val;
for (1..10) {
	my $offset = int rand $N_elements;
	Transfer($dev_ptr + $offset * $sizeof_float => $test_val);
	ok(unpack('f', $test_val) == $offset+1, "Position $offset has value " . ($offset+1));
}

# Run the multiply kernel and test 10 more random values.
CUDA::Min::Tests::cuda_multiply_by_constant($dev_ptr, $N_elements, 4);
for (1..10) {
	my $offset = int rand $N_elements;
	Transfer($offset * $sizeof_float + $dev_ptr => $test_val);
	ok(unpack('f', $test_val) == ($offset+1)*4, "Position $offset has value " . ($offset+1)*4);
}

# Return the values to their original state:
CUDA::Min::Tests::cuda_multiply_by_constant($dev_ptr, $N_elements, 0.25);

# Test the sum:
CUDA::Min::Tests::sum_reduce_test($host_array, $dev_ptr, 'a single block');
CUDA::Min::Tests::sum_reduce_test($host_array, $dev_ptr, '32 blocks', 32);

# Free the current device memory before moving forward:
Free($dev_ptr);
# Make sure that Free set $dev_ptr to zero:
ok($dev_ptr == 0, "Free sets freed memory pointers to zero");

# Now I'm going to try a nontrivial sum of 65536 floats:
$host_array = pack 'f*', map {sin ($_/10)} 1..65536;
$dev_ptr = MallocFrom($host_array);
CUDA::Min::Tests::sum_reduce_test($host_array, $dev_ptr, 'single block sine-sum');
CUDA::Min::Tests::sum_reduce_test($host_array, $dev_ptr, '32 block sine-sum', 32);
CUDA::Min::Tests::sum_reduce_test($host_array, $dev_ptr, '1024,32 block sine-sum', 1024, 32);

# Don't forget to clean up:
Free($dev_ptr);

