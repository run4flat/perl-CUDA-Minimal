=head1 NAME

CUDA::Min::Tests - a collection of tests for CUDA::Min.

=head1 USAGE

This provides a couple of CUDA kernels used in the CUDA::Min test suite.
You probably won't have need for any of these, unless you want to take a look at
the XS code in Test.xs for an idea of how to wrap CUDA kernel calls.

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2010 by David Mertens

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.10.1 or,
at your option, any later version of Perl 5 you may have available.

=cut

package CUDA::Min::Tests;

use 5.010001;
use strict;
use warnings;
BEGIN {
	require Test::More;
}

our $VERSION = '0.01';

require XSLoader;
XSLoader::load('CUDA::Min::Tests', $VERSION);

use CUDA::Min;
use Carp 'croak';

#####################
# Utility Functions #
#####################

# Computes the percent difference between two values:
sub percent_diff {
	my ($a, $b) = @_;
	return abs($a - $b) / (0.5 * abs($a + $b));
}

# Boolean function that determines if the input is a power of 2:
sub is_a_power_of_2 {
	local $_ = $_[0];
	return 0 if ($_ <= 0);			# negatives and zero are not allowed
	return 1 if ($_ == 1);			# 1 is considered a power of 2
	return 0 if ($_ % 2 == 1);		# odd numbers are not multiples of 2
	return is_a_power_of_2($_/2);	# check next power
}

###################
# Kernel Wrappers #
###################

# The Perl function that performs the actual reduction, using the CUDA kernel
# wrapper _cuda_sum_reduce. The Perl portion performs all the data validation
# and calls the cuda wrapper once for each element of @N_blocks, as well as for
# a blocksize of 1 for the final round.
sub cuda_sum_reduce {
	my ($dev_ptr, $N_bytes, @N_blocks) = @_;
	# Determine the number of elements in the array. We will eventually need to
	# copy a single float from the device, and this also serves to tell me how
	# large (in bytes) is a floating point number:
	my $result = pack('f', 1.1);
	my $length = $N_bytes / length($result);
	
	push @N_blocks, 1;
	
	# Validate the numbers of blocks for each round. All of them must be
	# multiples of the next, and all of them must be at least 32 times as large
	# as the next.
	for (1..$#N_blocks) {
		croak("Bug in test suite: each blocksize must be a multiple of next blocksize\n")
			unless $N_blocks[$_-1] % $N_blocks[$_] == 0;
		croak("Bug in test suite: each blocksize must be at least "
			. MIN_PER_BLOCK() . " times larger than the next blocksize\n")
			unless $N_blocks[$_-1] >= MIN_PER_BLOCK() * $N_blocks[$_];
	}
	# Also, the total length must be a multiple of the first blocksize, and it
	# must be at least 32 times as large as the first blocksize.
	croak("Bug in test suite: the length must be a multiple of the first blocksize\n")
		unless $length % $N_blocks[0] == 0;
	croak("Bug in test suite: the length must be at least " . MIN_PER_BLOCK()
		. " times larger than the largest blocksize\n")
		unless $length / $N_blocks[0] >= MIN_PER_BLOCK();
	# The first blocksize cannot be larger than 65536:
	croak("Bug in the test suite: the first blocksize cannot be larger than 65536\n")
		unless $N_blocks[0] <= 65536;
	# Last check: the last blocksize must be a multiple of 32 (which implicitly
	# forces all the previous blocksizes to be multiples of 32, also):
	croak("Bug in test suite: the length of the last block must be a multiple of "
		. MIN_PER_BLOCK() . "\n")
		if (@N_blocks > 1 and $N_blocks[-2] % MIN_PER_BLOCK() != 0);
	
	# At this point I know I can do at least one round of the reduction. I must
	# allocate the temporary memory first:
	my $dev_temp = Malloc(length($result) * $N_blocks[0]);
	
	# Keep calling this until we've run the sum with only a single block:
	foreach my $blocksize (@N_blocks) {
		# Make sure any previous calculations have finished, and make sure the
		# the kernel launch succeeded:
		ThreadSynchronize;
		GetLastError;
		_cuda_sum_reduce($dev_ptr, $length, $blocksize, $dev_temp);
		
		# Set the new length
		$length = $blocksize;
		
		# Make sure that all rounds after the first work on reducing the
		# temporary results, not the original results:
		$dev_ptr = $dev_temp;
	}
	ThreadSynchronize;
	GetLastError;
	
	# Clean up and return the final results:
	Transfer($dev_temp => $result);
	Free($dev_temp);
	return unpack ('f', $result);
}

# Perl function that invokes the XS wrapper for the multiply kernel. This
# performs the loan bit of data validation before calling the wrapper.
sub cuda_multiply_by_constant {
	my ($dev_ptr, $N_elements, $constant) = @_;
	
	# Validate the length:
	croak("Bug in test suite: Length must be a power of 2\n")
		unless is_a_power_of_2($N_elements);
	
	_cuda_multiply_by_constant($dev_ptr, $N_elements, $constant);
	ThreadSynchronize;
	GetLastError;
}

#################
# Test Function #
#################

# The Perl function that runs the gold (host-based) function on the data and
# compares the results with the CUDA-based function.
sub sum_reduce_test {
	# It's considered good practice to unpack your arguments. However, $_[0]
	# could potentially point to a huge string, which I don't want copied, so
	# I am not going to unpack that one.
	my ($dev_ptr, $description, @N_blocks) = @_[1..$#_];
	# The gold calculations perform the sum on the array in $_[0]:
	my $Gold_results = sum_reduce_gold($_[0]);
	# The CUDA calculations perform the sum on the device-allocated data:
	my $cuda_results = cuda_sum_reduce($dev_ptr, length($_[0]), @N_blocks);
	Test::More::diag "Gold gave $Gold_results and CUDA gave $cuda_results\n";
	Test::More::ok(percent_diff($Gold_results, $cuda_results) < 0.01, "Gold and CUDA sums for $description agree");
}

1;

