# Tests for the PDL functionality of CUDA::Minimal

use strict;
use warnings;
use Test::More;
use CUDA::Minimal;

# Put this in a begin block so that I don't get to the NiceSlice include
# if PDL is not on their machine.
BEGIN {
	eval 'use PDL';
	if ($@) {
		plan skip_all => 'PDL must be installed for the PDL tests';
	}
	else {
		plan tests => 15;
	}
}
use PDL::NiceSlice;

##########################
# Utility function works #
##########################

my $data = sequence(50);
my $slice = $data(0:9);
ok(CUDA::Minimal::_piddle_is_mmap_or_slice($slice),
	"Utility function correctly identifies slice");
ok(!CUDA::Minimal::_piddle_is_mmap_or_slice($data),
	"Utility function correctly identifies non-slice");

use PDL::IO::FastRaw;
my $test_fname = 'test.binary.data';
$data->writefraw($test_fname);

SKIP:
{
	my $mmap = eval {mapfraw $test_fname};
	skip('because PDL does not support mmap for your system', 1) if $@;
	ok(CUDA::Minimal::_piddle_is_mmap_or_slice($mmap),
		"Utility function correctly identifies mmap");
}

# remove the temporary file:
unlink $test_fname;
unlink "$test_fname.hdr";

# Test direct dataref updates:
my $negative_ones = -ones(10);
substr(${$data->get_dataref}, 0, PDL::Core::howbig($data->get_datatype)*10,
	${$negative_ones->get_dataref});
my $results = sequence(50);
$results(0:9) .= -1;
ok(all($data == $results), 'Direct data updates work');

# Data should still not be considered a slice
ok(!CUDA::Minimal::_piddle_is_mmap_or_slice($data),
	"Utility function still correctly identifies non-slice");


###############
# Basic Tests #
###############

# Generate some data (20 sequential doubles)
my $length = 20;
$data = sequence($length);

# Check that data allocation doesn't croak:
my $dev_ptr = eval {Malloc($data) };
ok($@ eq '', "Memory allocation from a piddle works");

# Copy it to the device:
eval {Transfer($data => $dev_ptr) };
ok($@ eq '', "Memory transfers from a piddle do not croak");

# Make new memory and make sure the two piddles do not compare well:
my $new_data = zeroes($length);
ok(not (all ($new_data == $data)), "Piddles do not initially agree");

# Copy the device data back to the new location:
eval {Transfer($dev_ptr => $new_data) };
ok($@ eq '', "Memory transfers to a piddle do not croak");
diag ($@) if $@;

# Compare the results:
ok(all ($new_data == $data), "Memory transfer did not copy garbage");

##############################
# copies to/from slices work #
##############################
$slice = $data(2:5);

# Try copying elements 2-5 from $data to the start of the device memory:
eval {Transfer($slice => $dev_ptr) };
ok($@ eq '', "Memory transfer from a slice does not croak");

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

# Copy data back to a slice:
$should_be = sequence(4) + 2;
$data(12:15)->get_from($dev_ptr);
# See if the data successfully transfered:
ok(all($data(12:15) == $should_be), "Transfer to a slice works")
	or diag("data was " . $data(12:15) . " and it should have been $should_be");

# working here - add croak tests for get_from and send_to, which fail at
# the moment

# There once was a time when a second transfer to a piddle failed.
# This tests that:
eval {Transfer($dev_ptr => $data(12:15))};
ok($@ eq '', "Second transfer to pdl does not croak")
	or diag($@);

# Clean up:
Free($dev_ptr);

