package CUDA::Min;

use 5.010001;
use strict;
use warnings;
use bytes;
use Carp 'croak';

# Needed to distinguish refs from objects
use attributes 'reftype';

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
		Sizeof CheckForErrors
	) ],
);

our @EXPORT_OK = ( @{ $EXPORT_TAGS{'all'} } );

our @EXPORT = qw(
	Free Malloc MallocFrom Transfer ThreadSynchronize GetLastError CheckForErrors
	Sizeof
);

our $VERSION = '0.01';

#######################
# Perl-side functions #
#######################

# an internal boolean routine that checks if the argument is an object:
sub is_an_object ($) {
	return ref($_[0]) and ref($_[0]) ne reftype($_[0])
}

# Just a wrapper for the _free function call, the 'internal' function that
# actually calls cudaFree.
sub Free {
	my @errors;
	# If called from cleanup code (from a previous croak or die), the caller
	# needs to save $@ on its own:
	$@ = '';
	
	# Run through all of the arguments and free them:
	my $i = 0;
	POINTER: for(@_) {
		# Now $_ refers to the actual passed scalar, which is supposed to be a
		# dev pointer.
		
		# Use these to capture troubles with free, but still work on the
		# rest of the device memory:
		eval {
			if (is_an_object $_) {
				# If it's an object that knows how to free itself, let it free
				# itself:
				if ($_->can('free_dev_memory')) {
					$_->free_dev_memory;
					next POINTER;
				}
				# If it's an object that at least knows about device memory,
				# retrieve that and free it.
				elsif ($_->can('get_dev_ptr')) {
					_free($_->get_dev_ptr);
				}
				# Otherwise, it's an object I don't know how to work with. Throw
				# an error (which is caught 10 lines below)
				else {
					die("Argument $i in the list looks like an object of type "
							. ref($_)
							. ", which does not appear to mimic device memory\n");
				}
			}
			# If it looks like a scalar, then just call _free on it:
			else {
				_free($_);
			}
			$i++;
		};
		# Collect all the errors, which I will report at the end:
		if ($@) {
			# Clean up any line references since I will be re-croaking the
			# error, with an updated line reference:
			$@ =~ s/ at .*$//;
			push @errors, $@;
			$@ = '';
		}
	}
	# Croak if there are errors:
	if (@errors) {
		croak(join("\n", @errors));
	}
}

# A wrapper for the _malloc function that checks if an argument is an object.
# If so, it gets the length and sends that value to _malloc:
sub Malloc ($) {
	croak('Cannot call Malloc in a void context') unless defined wantarray;
	# Special object processing:
	if (is_an_object $_[0]) {
		my $object = shift;
		if (my $func = $object->can('nbytes')) {
			return _malloc($func->($object));
		}
		else {
			croak('Attempting to allocate memory from an object of type '
				. ref($object) . ' but that class does not know how to say nbytes');
		}
	}
	# otherwise, just call _malloc, which will assume it's a packed string and
	# go from there.
	return _malloc($_[0]);
}

# A nice one-two punch to allocate memory on the device and copy the values
sub MallocFrom ($) {
	croak('Cannot call MallocFrom in void context') unless defined wantarray;
	my $dev_ptr = Malloc($_[0]);
	Transfer($_[0] => $dev_ptr);
	return $dev_ptr;
}

# A function that wraps the _transfer function, or delegates the transfer if it
# gets an object as one (or both) of the arguments:
sub Transfer ($$;$) {
	# If the first argument is an object and it mimics device memory, get the
	# device memory and call Transfer again:
	if (is_an_object($_[0]) and $_[0]->can('get_dev_ptr')) {
		my $object = shift;
		unshift @_, $object->get_dev_ptr;
		goto &Transfer;
	}
	# Repeat for the second argument:
	if (is_an_object($_[1]) and $_[1]->can('get_dev_ptr')) {
		splice @_, 1, 1, $_[1]->get_dev_ptr;
		goto &Transfer;
	}
	
	# If I'm here, I know that any mimicking of the device memory should have
	# been handled. Now I look to see if anything mimics host memory and call
	# that object's send_to function:
	if (is_an_object($_[0])) {
		my $send_to_func = $_[0]->can('send_to');
		croak("First argument to Transfer is an object, but it doesn't mimic either device or host memory")
			unless $send_to_func;
		goto &$send_to_func;
	}
	if (is_an_object($_[1])) {
		my $get_from_func = $_[1]->can('get_from');
		croak("Second argument to Transfer is an object, but it doesn't mimic either device or host memory")
			unless $get_from_func;
		# Rearrange the elements before going to the function:
		unshift @_, splice (@_, 1, 1);
		goto &$get_from_func;
	}
	
	# If I've gotten here, then none of the arguments are objects, they're all
	# scalars. I can just go to the underlying _transfer function, which will
	# handle everything else from here:
	goto &_transfer;
}

# A function that determines the number of bytes to which its argument refers.
# Sizeof can be an object that mimics host memory, or it can be a packed scalar,
# or it can be a spec string.
sub Sizeof ($;$) {
	# If it's an object, return the value of its nbytes method, or croak if
	# no such method exists:
	if (@_ == 1 and is_an_object($_[0])) {
		my $obj = shift;
		return $obj->nbytes if $obj->can('nbytes');
		croak("Argument to Sizeof is an object of type " . ref($obj)
					. ", but it does not mimic host memory");
	}
	# Otherwise if they supplied only one argument, assume it's a packed string
	# and just return the number of bytes:
	elsif (@_ == 1) {
		return length $_[0];
	}
	# Otherwise they supplied two arguments, so process it like a size spec.
	$@ = '';
	my ($type, $number) = @_;
	$number =~ /^\d+$/
		or croak("Bad size spec: second argument ($number) must be a number");
	# Compute the sizeof, in an eval, so if the pack string is bad, I can say so
	my $sizeof = eval{ length pack $type, 5 };
	return $number * $sizeof unless $@;
	
	# If pack croaked, send an error saying so:
	croak("Bad pack-string '$type' in size specifiation") if $@;
}

# A function to ensure that the given scalar has the correct length
sub SetSize ($$;$) {
	# Unpack the length. If it's a Sizeof spec, then pop off both arguments
	# and call Sizeof to get them; otherwise, assume the single argument is
	# the length in bytes:
	my $new_length = pop;
	if (@_ == 2) {
		$new_length = Sizeof($_[1] => $new_length);
	}

	# Delegate object methods:
	if (is_an_object($_[0])) {
		my $object = shift;
		if ($object->can('n_bytes')) {
			return $object->n_bytes($new_length);
		}
		else {
			croak("SetSize called on an object of type " . ref($object)
				. ", but it does not mimic host memory");
		}
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
	
	# Finish by returning the new length, in bytes:
	return $new_length;
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



####################
# PDL OO Interface #
####################

sub PDL::nbytes {
	my $self = shift;
	my $new_length = shift;
	croak("Changing the size of a piddle with SetSize or nbytes is not implemented")
		if (defined $new_length);
	
	return PDL::Core::howbig($self->get_datatype) * $self->nelem;
}

sub PDL::send_to {
	my ($self, $dev_ptr, ) = @_;
	
	Transfer(${$self->get_dataref} => $dev_ptr);
	$self->upd_data;
	return $self;
}

sub PDL::get_from {
	my ($self, $dev_ptr) = @_;
	Transfer($dev_ptr => ${$self->get_dataref});
	$self->upd_data;
	return $self;
}

1;
__END__

=head1 NAME

CUDA::Min - A minimal set of Perl bindings for CUDA.

=head1 SYNOPSIS

 use CUDA::Min';
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

=head1 PDL SYNOPSIS

 # The order of inclusion is not important:
 use PDL;
 use CUDA::Min;
 use PDL::NiceSlice;
 
 # Create a double array of 20 sequential elements:
 my $data = sequence(20);
 
 # Allocate memory on the device and transfer it:
 my $dev_ptr = MallocFrom($data);
 
 # Allocate more memory on the device with the same size:
 my $dev_ptr2 = Malloc($data);
 
 # Create some more values, host-side:
 my $randoms = $data->random;
 
 # Copy the random values to the device:
 Transfer($randoms => $dev_ptr2);
 
 # Using send_to PDL method:
 $randoms->send_to($dev_ptr2);
 
 # Copy the first five random values back
 # into elements 5-10 of the original array:
 $data(5:9)->get_from($dev_ptr2);
 
 # Free the device memory, of course:
 Free($dev_ptr, $dev_ptr2);

=head1 DESCRIPTION

This module provides what I consider to be a decent but minimal amount of
functionality to get Perl and CUDA talking together nicely, with an emphasis on
nicely. It works with plain-ol' packed Perl scalar, and it works nicely with
PDL. (However, it does not require that you have PDL installed to use it.)

It does not try to wrap all possible CUDA-C functions. I said nice, not
complete.

The functions provide basic methods for doing the following:

=over

=item allocating and deallocating Perl and device memory

L</MallocFrom> and L</Malloc> create memory on the device, L</Free> frees
device-side memory, and L</SetSize> ensures that your Perl scalar has the exact
amount of memory you want allocated

=item copying data to and from the device

L</Transfer> handles transfers to/from the device as well as between two memory
locations on the device, and two PDL methods, L</send_to> and L</get_from>,
allow for compact data transfer between piddles and device memory and standard
chaining syntax common to PDL

=item index manipulation

L</Offset> calculates pointer offsets from a given pointer value for you, and
L</Sizeof> is supposed to determine numbers of bytes in a safe way

=item thread synchronization

L</ThreadSynchronize> ensures that your kernel invocations have returned, which
is important for benchmarking

=item error checking

L</GetLastError> and L</CheckForErrors> provide methods for checking on and
getting the last errors; also, all function calls except L</ThreadSynchronize>,
L</GetLastError>, and L</CheckForErrors> croak when they encounter an error

=back

This module does not, however, provide any user-level kernels. (It does have one
kernel, for summing data, but don't use it; it's not very well written.) It is
hoped this will provide a base for managing data, so that later CUDA modules
can focus on writing kernels.

=head1 FUNCTIONS

=head2 MallocFrom (packed-array or piddle)

A convenient function for allocating and initializing device-side memory. It
will often be the case that you want to create an array on the device simply so
you can copy the contents of a host-side array. This function does both and
returns a scalar holding the pointer to the location in the device memory,
represented as an integer.

=over

=item Input

a piddle or a packed Perl scalar with data that you want copied to the device

=item Output

a scalar integer holding the pointer location of newly created, newly
initialized device memory

=back

To just create memory on the device without copying it, see L</Malloc>. This is
just a wrapper for L</Malloc> and L</Transfer>, so if you get any errors, see
those functions for error-message details.

=head2 Malloc (bytes or packed-array or piddle)

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

an integer with the number of desired bytes,
or a packed Perl scalar of the desired length,
or a piddle with the desired number and type of elements

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
 
 # PDL example:
 my $data = sequence(20);
 my $dev_ptr = Malloc($data);

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
And of course, a piddle obviously looks like a piddle.

=over

=item Input

a Perl scalar with the data source, a Perl scalar with the destination, and an
optional number of bytes

=item Output

none, although if the destination is a packed array or piddle, its contents will
be modified; if the destination is a chunk of device memory, its contents will
be modified

=back

You can get a number of errors with this. First, if you provide two device
pointers but don't specify the length, you'll get this:

 You must provide the number of bytes for device-to-device transfers

Also, you cannot use C<Transfer> to copy one Perl scalar to another:

 Transfer requires at least one of the arguments to be a device pointer
 but it looks like both are host arrays

You'll get that if both of your arguments is a packed scalar, or if one is a
packed scalar and the other is a piddle. If both are a piddle, you'll see this:

 You cannot call Transfer on two piddle values

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
desired length. This will not change the size of a piddle, and will croak if you
try to use it like that. C<SetSize> should only be called on packed scalars.

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

If you try to call this on a piddle, you will get the following error:

 Cannot call SetSize on a piddle.

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

=head1 PDL METHODS

CUDA::Min supports using piddles in place of packed scalars in most of its
arguments (everywhere you're likely to want them, at least) and it provides two
PDL methods for data transfer, L</send_to> and L</get_from>. Both of them take
only one argument: the device memory location.

=head2 send_to

Use this method to send the contents of a piddle to device memory. For an
example, see the L</PDL Example>.

=head2 get_from

Use this method to retrive the contents of a section of device memory I<and
store them in the invoking piddle>. You can think of this operation as happening
I<in place>.

=head2 PDL Example

This simple example loads a series of data sets from your disk and runs them
through a kernel (or collection of kernels) on your device. Don't be naive.
There is a lot of memory transfer happening here, and unless you have a I<lot>
of processing on the device, this is not going to be very efficient.

working here - this doesn't actually use the PDL methods

 use PDL;
 use PDL::IO::FastRaw;
 
 # Get the list of data files:
 my @files = glob('*.dat');
 
 # allocate memory for the results,
 # one double for each file:
 my $results = pdl(scalar(@files));
 
 # Allocate enough memory for the
 # results:
 my $dev_results = Malloc($results);
 
 # process the kernel and load the next file
 # while we wait for the kernel to finish
 forall my $file (@files) {
     # Load the file from disk:
     my $data = readfraw pop @files;
 
     # Push it to memory:
     my $dev_input = MallocFrom($data);
     
     # You'll have to write XS or Inline::C
     # code to make this function available
     # from Perl:
     invoke_my_handy_kernel($dev_input, $data->nelem
                   , $dev_results);
     
     # CUDA kernel invocations return
     # immediately, so make use of the time:
     $data = readfraw $file;
 }

=head1 Unspecified launch failure

Normally CUDA's error status is reset to C<cudaSuccess> after calling
C<cudaGetLastError>, which happens when any of these functions croak, or when
you call L</GetLastError> or L</CheckForErrors>. With one exception, later
checks for CUDA errors should be ok unless they actually had trouble. The
exception is the C<unspecified launch failure>, which will cause all further
kernel launches to fail with C<unspecified launch failure>. You can still copy
memory to and from the device, but kernel launches will fail. As far as I know,
the only way to clear that error is to quit and restart your process.

The best solution to this, in my opinion, is to make sure you have rock-solid
input validation before invoking kernels. If your kernels only know how to
process arrays that have lengths that are powers of 2, make sure to indicate
that in your documentation, and validate the length before actually invoking the
kernel. If your input to your kernels are good, this should not be a problem.

On the other hand, if you just can't find the bug in your invocation and need to
ship your code, you might be able to solve this with a multi-threaded approach,
or with forks. In that case you would have a parent process that spawns a child
process to invoke the kernels. If a kernel invocation went bad, the child could
pull all the important data off the device and save it in thread-shared host
memory, or to disk, and then die. If the child process ended prematurely, the
parent process could attempt to recover and spawn a new child process. However,
I do not have any multi-threaded tests in the test suite, so I can't even
promise that a multi-threaded approach would work. Best of luck to you if you
try this route, but I do not recommend it.

=head1 EXPORTS

This uses the standard L<Exporter> module, so it behaves in a fairly standard
way. I know it's considered modern to not export anything by default, but frakly
I think that's dumb for this module. As such, it exports the functions that I
used most when I was developing it, which include:

 :most 

If you're a purist and don't want those, there's nothing stopping you from
saying

 use CUDA::Min '';

which won't import anything. You can also use the C<:all> tag, which will import

 function-names

in addition to the ones already listed.

working here - a bit about combining :most with individual functions

You can also import individual functions by specifying their names. In that
case, you must fully qualify any functions that you don't import, such as
C<CUDA::Min::Sizeof> in this example:

 use CUDA::Min qw(Malloc Free);
 my $dev_ptr = Malloc(CUDA::Min::Sizeof '20f');
 ...
 Free($dev_ptr);

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

Finally, if you do anything numeric in Perl, you should look at L<PDL>.

=head1 AUTHOR

David Mertens, E<lt>dcmertens.perl.csharp@gmail.comE<gt>

That email address is obfuscated. My actual email address only has one
programming language in it. I'm sure you can figure out which one to remove.

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2010-2011 by David Mertens

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.10.1 or,
at your option, any later version of Perl 5 you may have available.

=head1 NEW API

You can now send objects in place of dev-ptr or packed-array.

=head2 packed-array object

A class that intends to mimc host memory needs to provide the following methods:

=over

=item nbytes

Getter and setter

=item send_to

Sends data to the device

=item get_from

Gets data from the device

=back

=head2 dev-ptr object

A class that intends to mimic device memory needs to provide the following
methods:

=over

=back

=item get_dev_ptr (required)

Returns an integer that can be interpreted as a device pointer

=item free_dev_memory (optional)

Frees device memory. If not supplied, the internal _free function will be
called on the result of get_dev_ptr, which will automatically set its argument
to zero.

=cut
