# Tests for the PDL functionality of CUDA::Min

use strict;
use warnings;
use Test::More;
use CUDA::Min;

# Put this in a begin block so that I don't get to the NiceSlice include
# if PDL is not on their machine.
BEGIN {
	eval 'use PDL';
	if ($@) {
		plan skip_all => 'PDL must be installed for the PDL tests';
	}
	else {
		plan tests => 8;
	}
}
use PDL::NiceSlice;

# Generate some data (20 sequential doubles)
my $length = 20;
my $data = sequence($length);

# Check that data allocation doesn't croak:
$@ = '';
my $dev_ptr = eval {Malloc($data) };
ok($@ eq '', "Memory allocation from a piddle works");
$@ = '';

# Copy it to the device:
eval {Transfer($data => $dev_ptr) };
ok($@ eq '', "Memory transfers from a piddle do not croak");
$@ = '';

# Make new memory and make sure the two piddles do not compare well:
my $new_data = zeroes($length);
ok(not (all ($new_data == $data)), "Piddles do not initially agree");

# Copy the device data back to the new location:
eval {Transfer($dev_ptr => $new_data) };
ok($@ eq '', "Memory transfers to a piddle do not croak");
diag ($@) if $@;
$@ = '';

# Compare the results:
ok(all ($new_data == $data), "Memory transfer did not copy garbage");

##############################
# copies to/from slices work #
##############################
my $slice = $data(2:5);

# Try copying elements 2-5 from $data to the start of the device memory:
eval {Transfer($slice => $dev_ptr) };
ok($@ eq '', "Memory transfer from a slice does not croak");
$@ = '';

# Verify that the copy did not sever the slice from the parent:
$slice .= -3;
ok(all($data(2:5) == -3), "Trasfer involving a slice did not sever the slice");

# At this point, the device data should contain the values
# 2, 3, 4, 5, 4, 5, 6, 7, ...
# Next, I confirm that.

# Copy all the values back to new_data:
Transfer($dev_ptr => $new_data);
# See if they are what they should be:
my $should_be = sequence($length);
$should_be(0:3) .= $should_be(2:5);
ok(all($new_data == $should_be)
	, "Transfer from slice modified only the memory it was supposed to");

# Try to Copy the revised data to the memory location

# Clean up:
Free($dev_ptr);

