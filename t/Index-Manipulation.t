# A set of tests for the index manipulation functions.

use Test::More tests => 7;
use CUDA::Min ':all';

# Check that Sizeof computes values correctly, and croaks on bad input:
ok(Sizeof(c => 20) == 20, "20 chars take 20 bytes");

eval{ Sizeof(20 => 'c') };
like($@, qr/Bad size spec: second argument/, "Sizeof croaks for bad spec string");
$@ = '';
eval{ Sizeof(0 => 20) };
like($@, qr/Bad pack-string/, "Sizeof croaks for bad pack type");
$@ = '';

# Create a packed array of 10 sequential doubles:
my $data = pack 'd*', 1..10;
my $length = length($data);

# Change that array so that it holds 20 doubles (the 10 extra are automatically
# set to zero):
SetSize($data, d => 20);
ok(length($data) == 2*$length, "SetSize corectly increases the length");

# Create a new array that's the same size as $data
my $new_data1;
SetSize($new_data1, length($data));
ok(length($new_data1) == length($data), "SetSize properly adjusts undef variables");

# or in one line:
SetSize(my $new_data2, length($data));
ok(length($new_data2) == length($data), "SetSize one-liner works");

# Shorten $new_data so it only holds 20 ints:
SetSize($new_data1, i => 20);
ok(length($new_data1) < length($new_data2), "SetSize correctly decreases the length");

