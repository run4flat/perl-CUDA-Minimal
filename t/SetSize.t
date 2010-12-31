# A set of tests for the SetSize function.

use Test::More tests => 4;
use CUDA::Min 'SetSize';

# Create a packed array of 10 sequential doubles:
my $data = pack 'd*', 1..10;
my $length = length($data);

# Change that array so that it holds 20 doubles (the 10 extra are automatically
# set to zero):
SetSize($data, 20, 'd');
ok(length($data) == 2*$length, "SetSize corectly increases the length");

# Create a new array that's the same size as $data
my $new_data1;
SetSize($new_data1, length($data));
ok(length($new_data1) == length($data), "SetSize properly adjusts undef variables");

# or in one line:
SetSize(my $new_data2, length($data));
ok(length($new_data2) == length($data), "SetSize one-liner works");

# Shorten $new_data so it only holds 20 ints:
SetSize($new_data1, 20, 'i');
ok(length($new_data1) < length($new_data2), "SetSize correctly decreases the length");

