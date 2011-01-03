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
		Offset Sizeof CheckForErrors
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

# A function that determines the number of bytes for a given string.
sub Sizeof ($) {
	my $spec = shift;
	$@ = '';
	if ($spec =~ /^(\d+)\s*(\w)$/) {
		my ($N, $pack_type) = ($1, $2);
		
		my $sizeof = eval{ length pack $pack_type, 0.45 };
		return $N * $sizeof unless $@;
	}
	# If pack croaked, send an error saying so:
	croak("Bad pack-string in size specifiation $spec") if $@;
	# If we're here, it's because the user's spec string didn't match the
	# required specification:
	croak("Size specification ($spec) must have the number of copies "
			. 'followed by the pack type');
}

# A little function that sorta does pointer arithmetic on device pointers:
sub Offset ($$) {
	my ($dev_ptr, $offset) = @_;
	# Make sure they sent in a meaningful offset value, which is either an
	# integer number of bytes or a Sizeof string:
	# string, which should have the
	# number of copies as the first part, and the pack type as the second:
	if ($offset =~ /^\d+$/) {
		return $dev_ptr + $offset;
	}
	return $dev_ptr + Sizeof($offset);
}

# A little function to ensure that the given scalar has the correct length
sub SetSize ($$) {
	# Unpack the length:
	my $new_length = $_[1];
	
	# Make sure the length is valid:
	$new_length = Sizeof($new_length) unless $new_length =~ /^\d+$/;
	
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

# Checks for errors. Returns the undefined value if there were no errors. If
# there were errors, it returns the error string, and also sets $@ to the same
# error string.
sub CheckForErrors () {
	my $error = GetLastError();
	return if $error =~ /no error/;
	$@ = $error;
	return $error;
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
 SetSize(my $results_array, length($host_data));
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
 
 # Copy -10 and -30 to the fifth and sixth elements:
 my $to_copy = pack('f*', -10, -30);
 Transfer($to_copy => Offset($input_dev_ptr, '4f'));

=head1 DESCRIPTION

This module provides what I consider to be the bare minimum amount of
functionality to get Perl and CUDA talking together nicely, with an emphasis on
nicely. It does not try to wrap all possible CUDA-C functions. It does not
attempt to talk with PDL or GSL. It works with plain-ol' packed Perl scalars
and passes around the device pointers as plain-ol' Perl integers.

The underlying assumption is that this will be used in conjunction with
Inline::C. In other words, if you need to do any host-side calculations with
your packed Perl scalar, you can write small C functions to do it quickly. Or
you can repackage your data in a PDL.

The functions provide basic methods for doing the following:

=over

=item allocating and deallocating Perl and device memory

L</MallocFrom> and L</Malloc> create memory on the device, L</Free> frees
device-side memory, and L</SetSize> ensures that your Perl scalar has the exact
amount of memory you want allocated

=item copying data to and from the device

L</Transfer> handles transfers to/from the device as well as between two memory
locations on the device

=item index manipulation

L</Offset> calculates pointer offsets from a given pointer value for you, and
L</Sizeof> is supposed to determine numbers of bytes in a safe way

=item thread synchronization

L</ThreadSynchronize> ensures that your kernel invocations have returned, which
is important for benchmarking

=item error checking

L</GetLastError> and L</CheckForErrors> provide methods for checking on and
getting the last errors; also, all the other function calls except
L</ThreadSynchronize> croak when they encounter an error

=back

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
intend to actually copy to the device. (For example, you need memory on the
device to hold the *output* of a kernel invocation.) To copy the contents of
your packed array, use L</MallocFrom>.

Like L</MallocFrom>, this returns a scalar holding the pointer to the
location in the device memory represented as an integer. Note that this does
not accept a sizeof-string, but you can call the L</Sizeof> function explicitly
if you want to use it.

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

 use CUDA::Min ':all';
 my $data = pack('f*', 1..10);
 my $dev_input_ptr = MallocFrom($data);
 my $dev_output_ptr = Malloc($data);
 # $dev_output_ptr now points to memory large
 # enough to hold as much information as $data
 
 # Allocate memory with an explicit number of bytes.
 # Holds 20 bytes:
 $dev_ptr1 = Malloc(20);
 # Holds 45 doubles:
 $dev_ptr2 = Malloc(Sizeof('45 d'));

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

One final note: C<Free> is not magical. If you actually copy the pointer address
to another variable, you will encounter trouble. This is most likely to crop
up if you store an offset in a seperate variable (see L</Offset>):

 # allocate device-side memory:
 my $dev_ptr = Malloc($double_size * $array_length);
 # get the pointer value to the third element:
 my $dev_ptr_third_element = Offset($dev_ptr, '2d');
 # do stuff
 ...
 # Free the device memory:
 Free $dev_ptr;
 # At this point, $dev_ptr is zero and
 # $dev_ptr_third_element points to a location
 # in device memory that is no longer valid.
 # Best to be safe and set it to zero, too:
 $dev_ptr_third_element = 0;

=head2 Transfer (source => destination, [bytes])

A simple method to move data around. The third argument, the number of bytes to
copy, is optional unless both the source and the destination are pointers to
memory on the device. If you don't specify the number of bytes you want copied
and either the source or the destination is a packed array, C<Transfer> will
copy an amount equal to the length of the array.

C<Transfer> determines which scalar is the host and which is the pointer to the
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

Also, you cannot use C<Transfer> to copy one Perl scalar to another:

 Transfer requires at least one of the arguments to be a device pointer
 but it looks like both are host arrays

If you try to copy more data to the host than it can hold, or if you try to copy
more data from the host than it currently holds (which only happens when you
specify a number of bytes and it's too large), you'll get this error:

 Attempting to transfer more data than the host can accomodate

The one remaining error happens when memory cannot be copied. It's an error
thrown by CUDA itself, and it's probably due to using an invalid pointer or
something. The error looks like this:

 Unable to copy memory: <reason>

=head2 Sizeof (sizeof-string)

Computes the size of the given specification. The specification has two parts.
The second part is the Perl L<pack> type (f for float, d for double, c for char,
etc), and the first is the number of copies you want. They can be seperated by
an optional space for clarity. Use this to calculate the number of bytes in
a way that makes your code clearer.

=over

=item Input

a specification string

=item Output

an integer number of bytes corresponding to the specification string

=back

If you have a bad specification string, this will croak with the following
message:

 Size specification (<specification>) must have the
 number of copies followed by the pack type

If you provided an invalid pack type, you'll see this:

 Bad pack-string in size specifiation <specification>

Here are some examples that will hopefully make things clear about specifying
good strings for C<Sizeof>:

 # the number of bytes in a 20-element double array:
 $bytes = Sizeof('20 d');
 # the number of bytes in an N-element character array:
 $bytes = Sizeof("$N c");

=head2 Offset (device-pointer, bytes or sizeof-string)

A somewhat slow and verbose but hopefully clear and forward-compatible method
for computing a pointer offset. Suppose you want to transfer data starting from
the fifth element of a device array of floats. You could hard-code like so:

 Transfer($dev_ptr + 20 => $result);

That uses a magic number. Use C<Offset> to clarify the meaning of your code:

 Transfer(Offset($dev_ptr => '5f') => $result)

You can simply specify the number of bytes, if you know them; otherwise, you
can specify a string that L</Sizeof> knows how to parse, and C<Offset> will get
the proper size that way.

Some examples should make this clearer:

 # Get the address of the fifth element of a double array:
 $dev_fifth = Offset($dev_ptr => '5d');
 # Get the offset of the nth element of a char array, where
 # the offset is stored in the variable $offset:
 $dev_offset_ptr = Offset($dev_ptr => "$offset c");
 # Since chars are always one byte, you could just specify
 # the number of bytes, like this:
 $dev_offset_ptr = Offset($dev_ptr => $offset);

=over

=item Input

the initial pointer (an integer) and an integer offset or a sizeof-string

=item Output

a pointer value offset from the original by the desired amount

=back

This does not throw any errors itself, but if you specify an erroneous
sizeof-string, L</Sizeof> will throw an error.

=head2 SetSize (perl-scalar, bytes or sizeof-string)

A function that ensures that the perl scalar that you're using has exactly the
length that you want. This function takes two arguments, the perl scalar whose
length you want set and the number of bytes or sizeof-string that specifies the
desired length.

=over

=item Input

a Perl scalar whose length is to be set and the new length

=item Output

none (it modifies the first argument directly)

=back

Here are some examples:

 # Create a packed array of 10 sequential doubles:
 my $data = pack 'd*', 1..10;
 # Change that array so that it holds 20 doubles (the 10 extra are automatically
 # set to zero):
 SetSize($data, '20d');
 # Create a new array that's the same size as $data
 my $new_data;
 SetSize($new_data, length($data));
 # or in one line:
 SetSize(my $new_data, length($data));
 # Shorten $new_data so it only holds 20 ints:
 SetSize($new_data, '20 i');

=head2 ThreadSynchronize

This is a simple wrapper for C<cudaThreadSynchronize>.

When you execute a kernel, the process returns immediately, before the kernel
finishes. This is useful if you want to do other things while the kernel runs
(some file IO during long-running kernels, for example). However, if you want to
benchmark your code, you need to be sure that the kernel has finished executing.
You accomplish this by copying data to/from the device (because those functions
wait for the threads to finish), or by calling this function. It takes no
arguments and returns nothing. It simply blocks until all the threads are done.
Unlike the other functions, it doesn't even croak if there was an error since
thread synchronization itself will never cause errors.

=head2 GetLastError

This is a simple wrapper for C<cudaGetLastError>. It returns a string describing
the last error. It returns exactly what CUDA tells it to say, so as of this time
of writing, if there were no errors, it returns the string 'no errors'. It also
clears the error so that further CUDA function calls will not croak. (But see
L</Unspecified launch failure> below.) For a slightly more Perlish error
checker, see L</CheckForErrors>.

=head2 CheckForErrors

Checks for CUDA errors. Returns the undefined value if there are no errors.
Otherwise, it returns the errors string. It also sets C<$@> with the error
string, if this was an error, so you'll want to be sure to clear that if you
find one. Note that this does not actually croak with the error; it simply sets
C<$@>.

=over

=item Input

none

=item Output

C<undef> if no errors or the error string if there were; also sets C<$@> with
the error string if there was one

=back

A nice idiom for error checking is this:

 if (CheckForErrors) {
     # Error handling here
     my $error_message = $@;
     # you could decide to croak with the error if you like:
     croak($@);
     # or you might decide to try to handle the error yourself:
     ... error handling code...
     # in which case you should finish by clearing the Perl error:
     $@ = '';
 }

Calling this function will clear CUDA's error status, if there was an error,
but see L</Unspecified launch failure> below.

=head1 Unspecified launch failure

Normally CUDA's error status is reset to C<cudaSuccess> after calling
C<cudaGetLastError>, which happens when any of these functions croak, or when
you call L</GetLastError> or L</CheckForErrors>. With one exception, later
checks for CUDA errors should be ok unless they actually had trouble. The
exception is the C<unspecified launch failure>, which will cause all further
kernel launches to fail with C<unspecified launch failure>. As far as I know,
the only way to clear that error is to quit and restart your process.

The best solution to this, in my opinion, is to make sure you have rock-solid
input validation before invoking kernels. If your input to your kernels are
good, and if they are invoked with good data, this should not be a problem.

On the other hand, if you just can't find the bug in your invocation and need to
ship your code, you might be able to solve this with a multi-threaded approach,
or with forks. In that case you would have a parent process that spawns a child
process to invoke the kernels. If a kernel invocation went bad, the child could
pull all the important data off the device and save it in thread-shared host
memory, and then die. If the child process ended prematurely, the parent process
could attempt to recover and spawn a new child process. However, I do not have
any multi-threaded tests in the test suite, so I can't even promise that a
multi-threaded approach would even work. :-)

=head1 EXPORTS

This exports no functions by default. If you like, you can choose to export all
functions. In other words, your code will look like this with no imports:

 use CUDA::Min;
 my $dev_ptr = CUDA::Min::Malloc(CUDA::Min::Sizeof '20f');

Or, you can choose to import all functions like this:

 use CUDA::Min ':all';
 my $dev_ptr = Malloc(Sizeof '20f');

You can also import individual functions by specifying their names:

 use CUDA::Min qw(Malloc Free);

=head1 BUGS AND LIMITATIONS

There is one big fat glaring bug with this library, which is that if you
allocate any memory on the device, the script will die, at the very very end,
with a segmentation fault. Any help on this would be much appreciated.

A potentially major shortcoming of this library is that it does not provide an
interface to global variables on the device. If have not decided if that's a bug
or a feature. I think it may be good because it requires kernel writers to
explicitly provide functions for manipulating global memory locations, which I
think is a good idea. Real-world usage will tell whether or not such a function
should be available.

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
programming language in it. I'm sure you can figure out which one to remove.

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2010-2011 by David Mertens

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.10.1 or,
at your option, any later version of Perl 5 you may have available.

=cut
