# A collection of tests for the Transfer function

use Test::More tests => 21;
use CUDA::Min;
use strict;
use warnings;

# Create a collection of values to play with:
my $N_elements = 1024;
my $host_array = pack ('f*', 1..$N_elements);

# Memory.t tested Malloc and Free, so I will assume those work.

# Test MallocFrom, which exercises Transfer:
$@ = '';
my $dev_ptr = eval{ MallocFrom($host_array) };
ok($@ eq '', 'MallocFrom works with reasonable data');
$@ = '';

# Get a scalar into which I can pull from the device
my $results = $host_array;

# Pull some random values and make sure they agree with what I expect
my $test_val = pack 'f', 1;
my $sizeof_float = length $test_val;

my $offset = int rand $N_elements;
# Test that Transfer only pulls a single value:
Transfer(Sizeof(f=>$offset) + $dev_ptr => $test_val);
ok(unpack('f', $test_val) == $offset+1, "Position $offset has value " . ($offset+1));

# Test that Transfer croaks when asked for more data than test_val can hold
eval{ Transfer($dev_ptr => $test_val, 200) };
like($@, qr/Attempting to transfer more data/
	, 'Transfer does not attempt to copy more data than the scalar can hold');
$@ = '';

# Test that scalar-to-scalar transfers are not allowed
eval{ Transfer($host_array => $test_val) };
like($@, qr/both are host arrays/
	, 'Transfers betwen two scalars is not allowed');
$@ = '';

# Test device-to-device transfers without a specified number of bytes:
eval{ Transfer($dev_ptr => $dev_ptr + Sizeof(f=>1)) };
like($@, qr/device-to-device transfers/
	, "Device-to-device transfers without a specified number of bytes croaks");
$@ = '';

# Test that you can specify the number of bytes for transfers involving scalars:
# Make an array filled with -3:
my $second_host_array = pack 'f*', map {-3} 1..20;
# Only copy the first five of those values:
eval{ Transfer($second_host_array => $dev_ptr, $sizeof_float * 5) };
ok($@ eq '', "Specifying the number of bytes for transfers with host doesn't croak");
$@ = '';
# Get the first ten of those values back:
eval{ Transfer($dev_ptr => $results, 10 * $sizeof_float) };
ok($@ eq '', "Specifying the number of bytes for transfers with host doesn't croak");
$@ = '';
# check that they are what they should be:
my $diff = 0;
my @results = unpack('f10', $results);
my @expected = qw(-3 -3 -3 -3 -3 6 7 8 9 10);
ok($results[$_] == $expected[$_], "Got expected result for entry $_") for (0..9);

# Test a device-to-device copy:
eval{ Transfer($dev_ptr => $dev_ptr + Sizeof(f=>30), $sizeof_float) };
ok($@ eq '', "Device-to-device transfers are OK");
# Test that it actually copied the correct value:
Transfer(Sizeof(f=>30) + $dev_ptr => $test_val);
ok(unpack('f', $test_val) == -3, "Device-to-device transfers work");
# Test that the next surrounding is OK:
Transfer(Sizeof(f=>29) + $dev_ptr => $test_val);
ok(unpack('f', $test_val) == 30, "Device-to-device transfers don't overwrite things");
Transfer(Sizeof(f=>31) + $dev_ptr => $test_val);
ok(unpack('f', $test_val) == 32, "Device-to-device transfers don't overwrite things");


# Clean up:
Free($dev_ptr);

