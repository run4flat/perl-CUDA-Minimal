package CUDA::Min;

use 5.010001;
use strict;
use warnings;
use bytes;
use Carp 'croak';

require Exporter;

our @ISA = qw(Exporter);

# Items to export into callers namespace by default. Note: do not export
# names by default without a very good reason. Use EXPORT_OK instead.
# Do not simply export all your public functions/methods/constants.

# This allows declaration	use CUDA::Min ':all';
# If you do not need this, moving things directly into @EXPORT or @EXPORT_OK
# will save memory.
our %EXPORT_TAGS = ( 'all' => [ qw(
		Free Malloc MallocFrom Transfer ThreadSynchronize GetLastError SetSize
	) ],
);

our @EXPORT_OK = ( @{ $EXPORT_TAGS{'all'} } );

our @EXPORT = qw(
	
);

our $VERSION = '0.01';

#######################
# Perl-side functions #
#######################

# Just a wrapper for the _free function call, the 'internal' function that
# actually calls cudaFree.
sub Free {
	eval { _free($_) foreach @_ };
	if ($@) {
		# Clean up the error and pass it along:
		$@ =~ s/ at .*$//;
		croak($@);
	}
}

# A nice one-two punch to allocate memory on the device and copy the values
sub MallocFrom ($) {
	croak('Cannot call MallocFrom in void context') unless defined wantarray;
	my $dev_ptr = Malloc($_[0]);
	Transfer($_[0] => $dev_ptr);
	return $dev_ptr;
}

# A little function to ensure that the given scalar has the correct length
sub SetSize ($$;$) {
	# Unpack the length:
	my $new_length = $_[1];
	if (@_ == 3) {
		# A third argument means that length refers to the number of elements,
		# and those elements are of a particular type. Use pack to get the
		# length of that type and multiply the length appropriately:
		$new_length *= length pack $_[2], 0;
	}
	
	# Make sure that we're working with an initialized value and get the length:
	$_[0] = '' if not defined $_[0];
	my $cur_length = length $_[0];

	# If it's already the right length, don't do anything:
	return if ($cur_length == $new_length);
	
	# If it's too short, use vec to increase the length:
	if ($cur_length < $new_length) {
		vec($_[0], $new_length - 1, 8) = 0;
		return;
	}
	
	# If it's too long, trim it back (I believe this is efficient, but I'm not
	# sure):
	substr ($_[0], $new_length) = '';
}

require XSLoader;
XSLoader::load('CUDA::Min', $VERSION);

1;
__END__

=head1 NAME

CUDA::Min - A minimal set of Perl bindings for CUDA.

=head1 SYNOPSIS

 use CUDA::Min ':all';
 # Create a host-side array of 10 sequential
 # single-precision floating-point values:
 my $N_values = 10;
 my $host_data = pack('f*', 1..$N_values);
 
 # Create memory on the device that holds a copy
 # of the scalar's data:
 my $input_dev_ptr = MallocFrom($host_data);
 
 # Create more memory on the device to hold the
 # results (assuming you need them to be in
 # seperate arrays on the device):
 my $results_dev_ptr = Malloc($host_data);
 
 # Be sure to free the device memory when you're done:
 END {
     Free($input_dev_ptr, $results_dev_ptr);
 }
 
 # Run your kernel on the data:
 my_fancy_kernel_invoker($input_dev_ptr
             , $results_dev_ptr, $N_values);
 
 # We would like to see the results, so copy them
 # back to the host in a newly allocated host array:
 SetSize(my $results array, length($host_data));
 # Copy the results back:
 Transfer($results_dev_ptr => $results_array);
 print "$_\n" foreach (unpack 'f*', $results_array);
 
 # Try running the inverse function and see if the
 # results agree with the original data. In his case,
 # I will be overwriting the original input memory:
 my_fancy_inverse_kernel_invoker($results_dev_ptr
             , $input_dev_ptr, $N_values);
 # Copy the final results back:
 Transfer($input_dev_ptr => $results_array);
 
 # Compare to the input data (there's probably a
 # faster way to do this, especially if you allow
 # yourself to use Inline::C, of if you use PDL):
 my $sq_diff = 0;
 my $sizeof_float = length pack 'f', 3.1415;
 for (my $i = 0; $i < length($host_data); $i += $sizeof_float) {
     $diff += (
                 unpack("x$i f", $host_data)
                 - unpack("x$i f", $results_array)
              )**2;
 }
 print "Round trip lead to an accumulated squared error of $sq_diff\n";

=head1 DESCRIPTION

This module provides what I consider to be the bare minimum amount of
functionality to get Perl and CUDA talking together nicely, with an emphasis on
nicely. It does not try to wrap all possible CUDA-C functions. It does not
attempt to talk with PDL or GSL. It works with plain-ol' packed Perl scalars.

The underlying assumption is that this will be used in conjunction with
Inline::C, so if you need to do anything to your packed Perl scalar, you can
write small C functions to do those.

=head1 FUNCTIONS

=head2 MallocFrom (packed-array)

A convenient function for allocating and initializing device-side memory. It
will often be the case that you want to create an array on the device simply so
you can copy the contents of a host-side array. This function does both and
returns a scalar holding the pointer to the location in the device memory,
represented as an integer.

=over

=item Input

a packed Perl scalar with data that you want copied to the device

=item Output

a scalar integer holding the pointer location of newly created, newly
initialized device memory

=back

To just create memory on the device without copying it, see L</Malloc>. This is
just a wrapper for L</Malloc> and L</Transfer>, so if you get any errors, see
those functions for error-message details.

=head2 Malloc (bytes or packed-array)

Unlike L</MallocFrom>, this can take either an integer with the desired number
of bytes to allocate, or a packed scalar with the length that you want to
allocate on the device. In other words, C<Malloc($N_bytes)> will allocate
C<$N_bytes> on the device, whereas C<Malloc($packed_array)> will allocate enough
memory on the device to hold the data in C<$packed_array>. The second form is
useful when you have a scalar of the appropriate length, but which you do not
intend to actually copy it to the device. (For example, you need memory on the
device to hold the output of a kernel invocation.) If you want to copy the
contents of your packed array, use L</MallocFrom>.

Like L</MallocFrom>, this returns a scalar holding the pointer to the
location in the device memory represented as an integer.

=over

=item Input

an integer with the number of desired bytes
or a packed Perl scalar of the desired length

=item Output

a scalar integer holding the pointer location of the device memory

=back

If this encounters trouble, it will croak saying:

 Unable to allocate <number> bytes on the device: <reason>

Usage example:

 use CUDA::Min ':simple';
 my $data = pack('f*', 1..10);
 my $dev_input_ptr = MallocFrom($data);
 my $dev_output_ptr = Malloc($data);
 # $dev_output_ptr now points to memory
 # large enough to hold $data

=head2 Free (device-pointer, device-pointer, ...)

Frees the device's memory at the associated pointer locations. You can provide
a list of pointers to be freed. If all goes well, this function finishes by
modifying your scalars in-place, setting them to zero. This way, if you call
C<Free> on the same perl scalar twice, nothing bad should happen.

=over

=item Input

Perl scalars with containing pointers obtained from L</Malloc> or L</MallocFrom>.

=item Output

none

=back

If this encounters trouble, it will croak saying:

 Unable to free memory on the device: <reason>

Good practice dictates that you should have a call to C<Free> for every call to
L</Malloc> or L</MallocFrom>. In order to keep yourself from forgetting about
this, for simple scripts I recommend putting your calls to C<Free> in C<END>
blocks immediately after your calls to L</Malloc>:

 my $position = Malloc($double_size * $array_length);
 my $velocity = Malloc($double_size * $array_length);
 END {
     Free $position, $velocity;
 }

=head2 Transfer (source => destination, [bytes])

A simple method to move data around. The third argument, the number of bytes to
copy, is optional unless both the source and the destination are pointers to
memory on the device. If you don't specify the number of bytes you want copied
and either the source or the destination is a packed array, Transfer will copy
an amount equal to the length of the array.

The function determines which scalar is the host and which is the pointer to the
device by examining the variables' internals. A packed array will always 'look'
like a character string, whereas a pointer will usually 'look' like an integer.

=over

=item Input

a Perl scalar with the data source, a Perl scalar with the destination, and an
optional number of bytes

=item Output

none, although if the destination is a packed array, its contents will be
modified

=back

You can get a number of errors with this. First, if you provide two device
pointers but don't specify the length, you'll get this:

 You must provide the number of bytes for device-to-device transfers

Also, you cannot use Transfer to copy one Perl scalar to another:

 Transfer requires at least one of the arguments to be a device pointer
 but it looks like both are host arrays

If you try to copy more data to the host than it can hold, or if you try to copy
more data from the host than it currently holds, you'll get this error:

 Attempting to transfer more data than the host can accomodate

The on remaining error happens when memory cannot be copied. It's an error
thrown by CUDA itself, and it's probably due to using an invalid pointer or
something. The error looks like this:

 Unable to copy memory: <reason>

=head2 SetSize (perl-scalar, new-length, [pack-type])

A function that ensures that the perl scalar that you're using has exactly the
length that you want. This function takes either two or three arguments. In the
two argument form, C<new-length> is taken as the length in bytes; in the three
argument form, C<pack-type> is a string indicating the L<pack> string type ('f'
for float, 'd' for double, etc) and C<new-length> is taken as the number of
elements of that type that you want. The pack type is not stored anywhere and it
has no impact on how you actually use the memory; it simply makes allocating
arrays of 10 floats or 20 ints a little bit clearer.

=over

=item Input

a Perl scalar whose length is to be set, the new length, and optionally the pack
type of the elements in the array.

=item Output

none (it modifies the first argument directly)

=back

Here are some examples:

 # Create a packed array of 10 sequential doubles:
 my $data = pack 'd*', 1..10;
 # Change that array so that it holds 20 doubles (the 10 extra are automatically
 # set to zero):
 SetSize($data, 20, 'd');
 # Create a new array that's the same size as $data
 my $new_data;
 SetSize($new_data, length($data));
 # or in one line:
 SetSize(my $new_data, length($data));
 # Shorten $new_data so it only holds 20 ints:
 SetSize($new_data, 20, 'i');

=head2 ThreadSynchronize

This is a simple wrapper for C<cudaThreadSynchronize>.

When you execute a kernel, the process returns immediately, before the kernel
finishes. This is useful if you want to do other things while the kernel runs
(some file IO during long-running kernels, for example). However, if you want to
benchmark your code, you need to be sure that the kernel has finished executing.
You accomplish this by copying data to/from the device, or by calling this
function. It takes no arguments and returns nothing. It simply blocks until all
the threads are done.

=head2 GetLastError

This is a simple wrapper for C<cudaGetLastError>, but usng Perl error handling
mechanisms.

This function checks for CUDA errors and croaks with an explanation if it
encounters one. Since all of the function calls in C<CUDA::Min> check for
errors, this may seem unnecessary. It's only real use is for checking if a
kernel launch failed. So, if any of the other C<CUDA::Min> functions are
mysteriously failing, try putting a C<ReportErrors> right before those function
calls to check for kernel launch trouble.

=head1 EXPORTS

This exports no functions by default. If you like, you can choose to export all
functions, in which case you'll get:

C<Free>, C<Malloc>, C<MallocFrom>, C<Transfer>, C<SetSize>, C<ThreadSynchronize>,
and C<GetLastError>.

=head1 BUGS AND LIMITATIONS

There is one big fat glaring bug with this library, which is that if you
allocate any memory on the device, the script will die, at the very very end,
with a segmentation fault. Any help on this would be much appreciated.

The major shortcoming of this library is that it does not provide an interface
to global variables on the device. That, probaly, should be addressed.

The library also does not provide commands to a myriad of function calls
provided by CUDA-C (i.e. the 'Array' and 'Async' functions, not to mention
stream, event, and thread management, OpenGL, Direct3D, and many others). Since
Kirk and Hwu (see references below) never use those functions, I have never used
them, and see no need to include them in a minimal set of bindings. Those sorts
of bindings would be more approriate for a L<CUDA::C> library, which I would
encourage others to consider writing. :-)

=head1 SEE ALSO

This module requires the nvcc compiler wrapper C<perl_nvcc> provided by
L<ExtUtils::nvcc>, and you will probably want to use C<perl_nvcc> to make Perl
wrappers for your kernels.

An entirely different approach to CUDA that you can leverage from Perl is
L<KappaCUDA>.

For general-purpose numerical computing using Perl, you should check out L<PDL>.
Linking PDL and CUDA should happen soon, but that is not the goal of this
library.

Some day soon, I hope to have created L<Inline::CUDA>, which would make wrapping
your CUDA kernels much simpler.

If you want to do anything quickly to your packed scalars on the host-side, and
you don't want to use PDL for some reason, consider using L<Inline::C>.

A decent book on the subject is David Kirk and Wen-mei Hwu's I<Programming
Massively Parallel Processors: A Hands-on Approach>.

=head1 AUTHOR

David Mertens, E<lt>dcmertens.perl.csharp@gmail.comE<gt>

That email address is obfuscated. My actual email address only has one
programming language in it. :-)

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2010 by David Mertens

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.10.1 or,
at your option, any later version of Perl 5 you may have available.

=cut
