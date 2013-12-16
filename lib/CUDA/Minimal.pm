package CUDA::Minimal;

use 5.008_008;
use strict;
use warnings;
use bytes;
use Carp;
use Scalar::Util qw(blessed);

require Exporter;

our @ISA = qw(Exporter);

our %EXPORT_TAGS = (
	'error' => [qw(ThereAreCudaErrors GetLastError PeekAtLastError)],
	'memory' => [qw(Free Malloc MallocFrom Transfer)],
	'util' => [qw(SetSize Sizeof)],
	'sync' => [qw(ThreadSynchronize)],
);

our @EXPORT_OK = ( @{ $EXPORT_TAGS{error} }, @{ $EXPORT_TAGS{memory} },
		@{ $EXPORT_TAGS{util}}, @{ $EXPORT_TAGS{sync} }
		);

our @EXPORT = ( @EXPORT_OK );

our $VERSION = '0.01';

#######################
# Perl-side functions #
#######################

=head1 NAME

CUDA::Minimal - A minimal set of Perl bindings for CUDA.

=head1 SYNOPSIS

 use CUDA::Minimal;
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
 my $sizeof_float = Sizeof f => 1;
 for (my $i = 0; $i < length($host_data); $i += $sizeof_float) {
     $diff += (
                 unpack("x$i f", $host_data)
                 - unpack("x$i f", $results_array)
              )**2;
 }
 print "Round trip lead to an accumulated squared error of $sq_diff\n";
 
 # I already freed the device memory 'earlier' by
 # putting it in an end block, but if you didn't do
 # that, you would put it here:
 Free($input_dev_ptr, $results_dev_ptr);
 

=head1 PDL SYNOPSIS

 # The order of inclusion is not important:
 use PDL;
 use CUDA::Minimal;
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
 $randoms->send_to($dev_ptr2);
 # the traditional function syntax works, too:
 Transfer($randoms => $dev_ptr2);
 
 # Copy the first five random values back
 # into elements 5-10 of the original array:
 $data(5:9)->get_from($dev_ptr2);
 
 # Free the device memory, of course:
 Free($dev_ptr, $dev_ptr2);

=head1 DESCRIPTION

This module provides what I consider to be a decent but minimal amount of
functionality to get Perl and CUDA talking together nicely, with an emphasis on
nicely. It works with plain-ol' packed Perl scalars, and it works nicely with
PDL. (However, it does not require that you have PDL installed to use it.)

It does not try to wrap all possible CUDA-C functions, and it's not even
up-to-date, even on the day of its release. I said nice, not
complete. These bindings were originally written months before CUDA Toolkit
4.0 was released, and although I've incorporated some new features, some of
them use old work-arounds. In fact, this does not even provide
any CUDA kernels, or even a framework for writing them.

Enough with the limitations of this module. The point is that B<IT WORKS>.
Furthermore, it has a nice interface and provides some degree of object
awareness, so that you can create object interfaces to device and host memory
and pass those objects to functions like C<MallocFrom>, C<Free>, and C<Transfer>.
If you would like to make objects that mimic device or host memory, see
L</Object Oriented Interfaces> below.

The functions in this module provide basic methods for doing the following:

=over

=item allocating and deallocating Perl and device memory

L</MallocFrom> and L</Malloc> create memory on the device, L</Free> frees
device-side memory, and L</SetSize> ensures that your Perl scalar has the exact
amount of memory you want allocated

=item copying data to and from the device

L</Transfer> handles transfers to/from the device as well as between two memory
locations on the device, and two PDL methods, L</send_to> and L</get_from>,
allow for concise data transfer between piddles and device memory using the
standard chaining syntax common to PDL

=item consistent size determination

Whether dealing with packed scalars, piddles, or some other object that handles
device memory, L</Sizeof> determines the number of bytes that variable can hold,
or allows you to compute the size from a succinct expression

=item thread synchronization

L</ThreadSynchronize> ensures that your kernel invocations have returned, which
is important for benchmarking

=item error checking

L</GetLastError>, L</ThereAreCudaErrors>, and L<PeekAtLastError> provide methods
for checking on and getting the most recent errors; also, all function calls
except L</ThreadSynchronize>, L</GetLastError>, L</ThereAreCudaErrors>,
and L</PeekAtLastError> croak when they encounter an error

=back

This module does not provide any user-level kernels. (It does have one
kernel, for summing data, but don't use it; it's not very well written.) It is
hoped this will provide a base for managing data, so that later CUDA modules
can focus on writing kernels rather than managing memory.

=head1 FUNCTIONS

=head2 MallocFrom (packed-array or piddle)

A convenient function for allocating and initializing device-side memory. It
will often be the case that you want to create an array on the device simply so
you can copy the contents of a host-side array. This function does both and
returns a scalar holding the pointer to the location in the device memory,
represented as an integer.

=over

=item Input

host-side memory with data that you want copied to the device

=item Output

a scalar integer holding the pointer location of newly created, newly
initialized device memory

=back

To just create memory on the device without copying it, see L</Malloc>.

If you don't have a variable that will hold the result of the function call,
C<MallocFrom> will croak saying

 Cannot call MallocFrom in void context

If you get other errors, see L</Malloc> and L</Transfer>, for which this
function is really just a wrapper.

=cut

# A nice one-two punch to allocate memory on the device and copy the values
sub MallocFrom ($) {
	croak('Cannot call MallocFrom in void context') unless defined wantarray;
	my $dev_ptr = Malloc($_[0]);
	Transfer($_[0] => $dev_ptr);
	return $dev_ptr;
}

=head2 Malloc (bytes or packed-array or host-side memory object)

In addition to a packed array or an object that manages host-side memory (which
are the only options for L</MallocFrom>), this function can also take an integer
with the desired number of bytes to allocate. (L</Sizeof> can be very helpful
for calculating the number of bytes based on type.) In other words,
C<Malloc($N_bytes)> will allocate C<$N_bytes> on the device, whereas
C<Malloc($packed_array)> will allocate enough memory on the device to hold the
data in C<$packed_array>. C<Malloc($host_memory_object)> uses the object's 
C<nbytes> function to determine how many bytes to allocate on the divice.

Allocating device memory based on the size of a packed scalar or memory object
is useful when you have a scalar of the appropriate length, but which you do not
intend to actually copy to the device. (For example, you need memory on the
device to hold the I<output> of a kernel invocation.) To copy the contents of
your packed array in tandem with the allocation, use L</MallocFrom>.

Like L</MallocFrom>, this returns a scalar holding the pointer to the
location in the device memory represented as an integer, and it will croak if
called in void context.

=over

=item Input

an integer with the number of desired bytes,
or a packed Perl scalar of the desired length,
or an object with the desired number and type of elements

=item Output

a scalar integer holding the pointer location of the device memory

=back

If this encounters trouble, it will croak saying either

 Cannot call Malloc in void context.

which means you're ignoring the return value (bad for memory leaks), or

 Attempting to allocate memory from an object of type
 <type> but that class does not know how to say nbytes

which means that you supplied an object that does not have the C<nbytes>
method. That means you're probably using the wrong object. Finally, you could
get this error:

 Unable to allocate <number> bytes on the device: <reason>

which happens when CUDA was unable to allocate memory on the device. (The reason
is CUDA's reason, not mine.)

Usage example:

 use CUDA::Minimal;
 my $data = pack('f*', 1..10);
 my $dev_input_ptr = MallocFrom($data);
 my $dev_output_ptr = Malloc($data);
 # $dev_output_ptr now points to memory large
 # enough to hold as much information as $data
 
 # Allocate memory with an explicit number of bytes.
 # Holds 20 bytes:
 $dev_ptr1 = Malloc(20);
 # Holds 45 doubles:
 $dev_ptr2 = Malloc(Sizeof(d => 45));
 
 # PDL example:
 my $data = sequence(20);
 my $dev_ptr = Malloc($data);

=cut

sub Malloc ($) {
	croak('Cannot call Malloc in a void context') unless defined wantarray;
	my $to_return = eval {
		# Special object processing:
		if (blessed $_[0]) {
			my $object = shift;
			if (my $func = $object->can('nbytes')) {
				return _malloc($func->($object));
			}
			else {
				croak('Attempting to allocate memory from an object of type '
					. blessed($object) . ' but that class does not know how to say nbytes');
			}
		}
		# otherwise, just call _malloc, which will assume it's a packed string and
		# go from there.
		
		return _malloc($_[0]);
	};
	
	return $to_return if defined $to_return;
	
	# If we've reached here, then something must be wrong. Clean up and
	# recroak:
	$@ =~ s/ at .*?\d\d\d\.\n$//;
	croak($@);
}

=head2 Free (device-pointer, device-pointer, ...)

Frees the device's memory at the associated pointer locations. You can provide
a list of pointers to be freed. If all goes well, this function finishes by
modifying your scalars in-place, setting them to zero. This way, if you call
C<Free> on the same perl scalar twice, nothing bad should happen. This also
accepts objects that mimic device memory.

=over

=item Input

Perl scalars containing pointers obtained from L</Malloc> or L</MallocFrom>,
or objects that mimic device memory.

=item Output

none

=back

This may croak for a few reasons. The most generic reason is this:

 Unable to free memory on the device: <reason>

This is a CUDA error, and usually is only a problem if you supplied a bad
memory location. Another reason for death would be:

 Argument <number> in the list looks like an object of type <type>,
 which does not appear to mimic device memory

which means you passed an object to C<Free> that does not know how to minic
device memory.

You may also get this warning:

 Argument <number> not defined

which means you sent and undefined value as an argument to C<Free>. It's
harmless, from the standpoint of C<Free>, but may indicate a bug in your code.

Good practice dictates that you should have a call to C<Free> for every call to
L</Malloc> or L</MallocFrom>. For simple scripts, you might want to consider
putting your calls to C<Free> in C<END> blocks immediately after your calls to
L</Malloc>:

 my $position = Malloc($double_size * $array_length);
 my $velocity = Malloc($double_size * $array_length);
 END {
     Free $position, $velocity;
 }

One final note: C<Free> is not magical. If you actually copy the pointer address
to another variable, you will encounter trouble.

 # allocate device-side memory:
 my $dev_ptr = Malloc(Sizeof(d => $array_length));
 # get the pointer value to the third element:
 my $dev_ptr_third_element = $dev_ptr + Sizeof(d => 2);
 # do stuff
 ...
 # Free the device memory:
 Free $dev_ptr;
 # At this point, $dev_ptr is zero and
 # $dev_ptr_third_element points to a location
 # in device memory that is no longer valid.
 # Best to be safe and set it to zero, too:
 $dev_ptr_third_element = 0;

=cut

# This used to be a rather trivial wrapper for _free, but that is no longer
# the case. At any rate, it is the public interface for memory deallocation.
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
			if (blessed $_) {
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
							. blessed($_)
							. ", which does not appear to mimic device memory\n");
				}
			}
			# If it looks like a scalar, then just call _free on it:
			elsif (not defined $_) {
				carp("Argument $i not defined\n");
			}
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

=head2 Transfer (source => destination, [bytes, offset])

A simple method to move data around. The third argument, the number of bytes to
copy, is optional unless both the source and the destination are pointers to
memory on the device. If you don't specify the number of bytes you want copied
and either the source or the destination is host memory, C<Transfer> will copy
as much data as the host memory can accomodate. The offset is optional, except
that it is not allowed for device-to-device transfers.

The use of the fat comma is not required, but is strongly encouraged.

C<Transfer> determines which scalar is the host and which is the pointer to the
device by examining the variables' internals. Objects are always blessed
references, a packed array will always 'look' like a character string, and a
pointer will usually 'look' like an integer.

=over

=item Input

the source, the destination, an optional number of bytes, an optional host
offset in bytes

=item Output

none, although the destination's data (host or device) will be modified

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

(You might complain and say, "Why doesn't it Do What I Mean and extend the
scalar to accomodate the length?" Such behavior could lead to very large yet
silent memory allocations, which I would rather avoid. But I might consider
changing this in the future, so you should not depend on the fact that it does
not modify the host memory. If you do, let me know and I'll take that into
consideration before changing anything.)

If you try to Transfer data from an object that does not know how to
C<get_dev_ptr> (mimic device memory) or C<send_to> (mimic host memory), you
will get the error

 First argument to Transfer is an object, but
 it doesn't mimic either device or host memory

Similarly, if you try to Transfer data to an object that does not know how to
C<get_dev_ptr> (mimic device memory) or C<get_from> (mimic host memory), you
will get the error

 Second argument to Transfer is an object, but
 it doesn't mimic either device or host memory

The one remaining error happens when memory cannot be copied. It's an error
thrown by CUDA itself, and it's probably due to using an invalid pointer or
something. The error looks like this:

 Unable to copy memory: <reason>

working here - new errors:

 Host offsets are not allowed for device-to-device transfers
 Host offset must be less than the host's length

=cut

# A function that wraps the _transfer function, or delegates the transfer if it
# gets an object as one (or both) of the arguments:
sub Transfer ($$;$$) {
	# If the first argument is an object and it mimics device memory, get the
	# device memory and call Transfer again:
	if ( blessed $_[0] and $_[0]->can('get_dev_ptr') ) {
		my $object = shift;
		unshift @_, $object->get_dev_ptr;
		goto &Transfer;
	}
	# Repeat for the second argument:
	if ( blessed $_[1] and $_[1]->can('get_dev_ptr') ) {
		splice @_, 1, 1, $_[1]->get_dev_ptr;
		goto &Transfer;
	}
	
	# If I'm here, I know that any mimicking of the device memory should have
	# been handled. Now I look to see if anything mimics host memory and call
	# that object's send_to function:
	if (blessed($_[0])) {
		my $send_to_func = $_[0]->can('send_to');
		croak("First argument to Transfer is an object, but it doesn't mimic either device or host memory")
			unless $send_to_func;
		goto &$send_to_func;
	}
	if (blessed($_[1])) {
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

=head2 Sizeof (object or specification)

Computes the size of the given object or specification. If you want to know the
number of bytes that a packed array has, use this function. If you want to know
the number of bytes in a 20-element integer array, use this function (with
different arguments). If you want to know the number of bytes available in
some object that mimics host-side memory, you could say C<< $object->nbytes >>,
or you could use this function as C<Sizeof($object)>. Basically, it's the
get-the-size catch-all function.

=over

=item Input

a packed array, or a host side memory object, or pack-type=>quantity pair of
arguments

=item Output

an integer number of bytes

=back

The usage for a packed array or host-side memory is hopefully straight-forward.
C<Sizeof($packed_array)> will give you the number of bytes in the array, and
similarly for an object. That is intentional, and hopefully means you can write
code that does not have to check if the argument is an object in order to
determine its size.

The specification is a little different. In that case, you give two arguments.
The first is the Perl L<pack> type (f for float, d for double, c for char,
etc), and the second is the number of copies you want. Such a usage looks like
this: C<< Sizeof(d => 20) >>, which returns the number of bytes in a 20-element
array of doubles. That's particularly useful when you only want to transfer a
certain amount of data, or when you need to allocate memory in
a packed array in preparation for a transfer from the device to the host
(using L</SetSize> together with this function, I would expect).

If you have a hard time remembering the order (is it type => quantity, or 
quantity => type?) just remember that the fat comma is intentional, and the
order allows you to write the Sizeof spec without using any quotes when you
C<use strict>. Compare:

 use strict;
 # WRONG order:
 my $bytes = Sizeof(20 => 'd');
 # RIGHT order:
 my $bytes = Sizeof(d => 20);

This may croak for a number of reasons. If you try to get the size of an object
but the object does not know how to say C<nbytes>, you will get this error:

 Argument to Sizeof is an object of type <class-name>,
 but it does not mimic host memory

That means that you are indeed using a Perl object, but it does not know how
many bytes it can hold.

On the other hand, if you specify a pack-type => quantity pair, you will get
the following error if the quantity does not look like a number:

 Bad size spec: second argument (<value>) must be a number

Whatever is inside the parenthesis is what Sizeof thinks you're trying to use as
a number, and Sizeof does not think it resembles a positive integer. Or you can
have trouble if you use an invalid pack-type (see Perl's documentation on the
L<pack> function):

If you have a bad specification string, this will croak with the following
message:

 Size specification (<specification>) must have the
 number of copies followed by the pack type

If you provided an invalid pack type, you'll see this:

 Bad pack-string '<type>' in size specifiation

The expression inside the quotes is what you provided. This will only trigger
an error if Perl's pack function actually croaks while parsing it, so
double-check your actual verion of Perl's documentation for pack if you get
trouble here.

Here are some examples that will hopefully make things clearer if you still
don't quite understand:

 # the number of bytes in a 20-element double array:
 $bytes = Sizeof(d => 20);
 # Allocate a packed-array with 30 ints:
 SetSize(my $output, Sizeof(i => 20));

=cut

sub Sizeof ($;$) {
	# If it's an object, return the value of its nbytes method, or croak if
	# no such method exists:
	if (@_ == 1 and blessed($_[0])) {
		my $obj = shift;
		return $obj->nbytes if $obj->can('nbytes');
		croak("Argument to Sizeof is an object of type " . blessed($obj)
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

=head2 SetSize (host-memory, bytes or Sizeof-spec)

A function that ensures that the object or Perl scalar that you're using has
exactly the length that you want. This function takes two or three arguments:
the host memory whose length you want set and the number of bytes or
the a Sizeof-compatible specification (type=>quantity) that specifies the
desired length.

I have written this so that if you call it on an object, it will call the
object's C<nbytes> function with an argument. That is, it will try to set the
object's length using the C<nbytes> function as a setter.

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
 # set to zero by Perl):
 SetSize($data, '20d');
 # Create a new array that's the same size as $data
 my $new_data;
 SetSize($new_data, length($data));
 # or in one line:
 SetSize(my $new_data, length($data));
 # Shorten $new_data so it only holds 20 ints:
 SetSize($new_data, '20 i');

=cut

sub SetSize ($$;$) {
	# Unpack the length. If it's a Sizeof spec, then pop off both arguments
	# and call Sizeof to get them; otherwise, assume the single argument is
	# the length in bytes:
	my $new_length = pop;
	if (@_ == 2) {
		$new_length = Sizeof($_[1] => $new_length);
	}

	# Delegate object methods:
	if (blessed($_[0])) {
		my $object = shift;
		if ($object->can('n_bytes')) {
			return $object->n_bytes($new_length);
		}
		else {
			croak("SetSize called on an object of type " . blessed($object)
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
the last error. It returns exactly what CUDA tells it to say, so as of the time
of writing, if there were no errors, it returns the string 'no errors'. It also
clears the error so that further CUDA function calls will not croak. (But see
L</Unspecified launch failure> below.) To get the name of the error without
resetting the error condition, see L</PeekAtLastError>. I you just want to know
if CUDA is in an error state, use L</ThereAreCudaErrors>.

=head2 PeekAtLastError

This is a simple wrapper for C<cudaPeekAtLastError>. It returns a string describing
the last error, as described in L<GetLastError>. Unlike the L<GetLastError>,
C<PeekAtLastError> does not clear the error.

=head2 ThereAreCudaErrors

I will admit, the name of this function is a bit odd.
I named it this way so that the following statements read as proper Enlish:

 if (ThereAreCudaErrors) {
     # Do some error handling.
 }

or this:

 return ($the_answer) unless ThereAreCudaErrors;
 # handle errors here and perhaps return something else

C<ThereAreCudaErrors> is essentially a boolean verion of L<PeekAtLastError> 
which returns true when an error has occurred and false otherwise.

Calling this function will not clear CUDA's error status, if there was an error,
but see L</Unspecified launch failure> below.

=over

=item Input

none

=item Output

A false value if no errors or a true value if there were

=back

=cut

sub ThereAreCudaErrors () {
	return PeekAtLastError() ne 'no error';
}

require XSLoader;
XSLoader::load('CUDA::Minimal', $VERSION);

##############################
# OBJECT ORIENTED INTERFACES #
##############################

=head1 Object Oriented Interfaces

The documentation made many references to objects that mimic device or host
memory. Objects that 'mimic' device or host memory have a handful of methods
that C<CUDA::Minimal> expects to be able to call on them.

=head2 Host memory

A class that intends to mimc host memory needs to provide the following methods:

=over

=item nbytes

When called without any arguments, this function should return the number of
bytes that the host memory supports. When called with an argument, the function
must take the argument as the new number of desired bytes and resize itself
accordingly, or croak if that is not possible. If called with an argument, it
should return the I<previous> number of bytes allocated.

=item send_to

This method should accept one or two arguments. The first is the device pointer
or an object that mimics device memory. The second is an optional argument that
states the number of bytes to copy. There is no required return value, though
the PDL method returns the just-used piddle to allow for further chaining. Such
is a common idiom in PDL.

=item get_from

Just as with C<send_to>, this method should accept one or two arguments. The
first is the device pointer (or an object that mimics device memory) and the
second is an optional number of bytes to copy to the object from the device.
You do not have to return anything from this function, thought the PDL method
returns the object that called it.

=back

=head2 Device Memory

A class that intends to mimic device memory needs to provide the following
methods:

=over

=item get_dev_ptr (required)

Returns an integer that can be interpreted as a device pointer.

=item free_dev_memory (optional)

Frees device memory. If not supplied, the internal _free function will be
called on the result of the C<get_dev_ptr> method, and should automatically set
its argument to zero. This function should croak if it fails (the croak will be
captured within L</Free> if it was called by L</Free>). Also, the return value
is ignored when called by L</Free>, so you can have it return whatever you like,
or nothing at all.

=back

=cut



####################
# PDL OO Interface #
####################

=head1 PDL METHODS

As an illustration of an object-oriented interface, CUDA::Minimal supports using
piddles in its arguments. It provides two
PDL methods for data transfer, L</send_to> and L</get_from>, and another method
for getting and setting the number of bytes, L</nbytes>. Both of the transfer
functions take between one and three arguments: the device memory location and,
optionally, the number of bytes to copy and the byte offset.

I recommend against setting the length of a piddle using L</nbytes>. See that
method's documentation for a discussion of why I think this is bad.

If you want to copy data to/from a portion of a piddle, I think it's clearer to
operate on a slice of a piddle rather than specifying the number of bytes to
copy. Slices let you work with arbitrary subsections of piddles, whereas
specifying the number of bytes and an offset only lets you work with a
contiguous segment of data from the piddle. However, operations on slices
make internal copies of the slice's data before transfering them, which can lead
to lots of extra memory allocations if you try to transfer a large slice of an
even large piddle to or from device memory. So, if you need to transfer a large
contiguous chunk of memory from an even larger piddle, it would be more efficient
to specify an offest and the number of bytes to copy rather than operating on a
slice.

Finally, it should be noted that, at the time of writing, it is not possible to
transfer data directly to/from mmapped data. Hopefully that will be resolved in
the future.

=head2 PDL Example

This simple example loads a series of data sets from your disk and runs them
through a kernel (or collection of kernels) on your device.
There is a lot of memory transfer happening here, and unless you have a I<lot>
of processing on the device, this is probably going to be very inefficient.

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

=head2 send_to

A method that sends the contents of a piddle to device memory.

For example,

 my $N_simulations = 20;
 my $N_members = 256;
 my $distribution = zeroes(float, $N_members);
 my $dev_distribution = Malloc($distribution);
 
 for (1..$N_simulations) {
     $distribution .= $distribution->grandom;
     
     $distribution->send_to($dev_distribution);
     
     my_statistics_kernel($dev_distribution, $N_members);
 }

In this example, you never actually need to know the distribution in your
Perl-side code, so you could instead create a template piddle with the type and
number of elements and use that to generate the random values for you on the fly:

 my $N_simulations = 20;
 my $N_members = 256;
 my $template = zeroes(float, $N_members);
 my $dev_distribution = Malloc($distribution);
 
 for(1..$N_simulations) {
     $template->grandom->send_to($dev_distribution);
     my_statistics_kernel($dev_distribution, $N_members);
 }

You can explicitly specify the number of bytes to copy using C<send_to>, if you
like. Alternatively, if you want to copy fewer bytes from your piddle
to device memory, simply slice out the data you want, and send that:

 use PDL::NiceSlice;
 $piddle(20:32)->send_to($dev_memory);

The only method-specific noise this will generate is a warning if you try to
send a magical piddle, in which case you will get the following warnings:

 Transfering data from magical (mmapped?) piddle may be inefficient

(At the time of writing, such warnings should only come from mmapped piddles or
piddles involved with special operations in PDL::Graphics::PLplot.) To avoid
this warning, send a slice of the piddle you wish to transfer, rather than the
piddle itself.

=cut

sub PDL::send_to {
	my $self = shift;
	my $dev_ptr = shift;
	
	eval{ Transfer(${$self->get_dataref} => $dev_ptr, @_) };
	
	# Check for problems with mmapped data and try a work-around:
	if ($@ =~ /^Trying to get dataref to magical/) {
		carp("Transfering data from magical (mmapped?) piddle may be inefficient");
		$@ = '';
		my $to_copy = $self->new;
		$to_copy .= $self;
		eval{ Transfer($dev_ptr, ${$to_copy->get_dataref}, @_) };
	}
	
	if ($@) {
		# If I encountered an error, clean up the message and recroak:
		$@ =~ s/ at .*Min.pm line \d+\.\n//;
		croak($@);
	}
	return $self;
}

=head2 get_from

Use this method to retrive the contents of a section of device memory I<and
store them in the invoking piddle>. You can think of this operation as happening
I<in place>.

Here's an example:

 my $results = zeroes(50);
 my $dev_results = Malloc($results);
 
 # Run your kernel:
 run_my_kernel($input_dev_pointer, $N_items, $dev_results);
 
 # Copy the results back:
 $results->get_from($dev_results);

Just like L</send_to>, you can apply this to a slice if you want. This will only
copy enough data to fill the piddle on which it is called, so even if the device
memory has enough memory for 200 elements, if the piddle only holds 20 elements,
this method will only copy 20 elements:

 # Allocate space for all the results:
 my $full_results = zeroes($N_runs, $N_data_points);
 
 # Allocate enough device memory for one run's worth of results:
 my $dev_results = Malloc($full_results(0,:));
 
 # Perform all the runs:
 for (1..$N_runs) {
     run_my_kernel($input_dev_pointer, $N_items, $dev_results);
     
     # Copy the results into the space allocated for this run:]
     $full_results($_, :)->get_from($dev_results);
 }

Alternatively, you can explicitly specify the number of bytes to copy as the
second argument to the methdod.

=cut

# A utility function that determines if the underlying piddle is an mmapped
# piddle or a slice:
sub _piddle_is_mmap_or_slice {
	my ($self) = @_;
	
	$@ = '';
	# Backup the current value:
	my $backup = $self->flat->at(0);
	eval {
		# Prepare the piddle for its test (set it strictly to zero):
		$self->flat->set(0, 0);
		# Prepare the alternative value (set it strictly to one):
		my $test_pdl = $self->zeroes($self->type, 1);
		$test_pdl->set(0, 1);
		
		# Perform the copy at the dataref level:
		substr (${$self->get_dataref}, 0, PDL::Core::howbig($self->get_datatype)
			, ${$test_pdl->get_dataref});
		
		# At this point, the mmapped piddles will have thrown an error. Since
		# the memory location of the dataref was not modified, I do not need to
		# call upd_data. A physical piddle will have the new value at its
		# zeroeth location; a slice will not:
		if ($self->flat->at(0) == 0) {
			die "Is a slice";
		}
	};

	# Undo any changes:
	$self->flat->set(0, $backup);
	
	if ($@) {
		$@ = '';
		return 1;
	}
	return 0;
}

sub PDL::get_from {
	my $self = shift;
	my $dev_ptr = shift;
	
	# Voice opposition to bad values
	Carp::cluck("nelem/length mismatch!")
		if $self->nelem * PDL::Core::howbig($self->get_datatype)
			!= length(${$self->get_dataref});
	
	# If I'm dealing with a piddle that knows how to respond to get_dataref
	# and will do what it's supposed to do, then just send the dataref to
	# Transfer:
	unless (_piddle_is_mmap_or_slice($self)) {
		eval { Transfer($dev_ptr, ${$self->get_dataref}, @_) };
		# If I encountered an error, clean up the message and recroak:
		if ($@) {
			$@ =~ s/ at .*Min.pm line \d+\.\n//;
			croak $@;
		}
		return $self;
	}

	# If we're here for whatever reason, I have to make a physical piddle with
	# the same dimensions as $self, into which I can perform the transfer. Make
	# a copy specifically to receive the data:
	my $to_copy = $self->new;
	
	# Transfer the data; catch, clean, and reissue any errors:
	eval { Transfer($dev_ptr, ${$to_copy->get_dataref}, @_) };
	if ($@) {
		$@ =~ s/ at .*Min.pm line \d+\.\n//;
		croak($@);
	}
	
	# Copy the results back to self and return self:
	$self .= $to_copy;
	return $self;
}

=head2 nbytes

A rather dumb method for changing the size of a piddle. Use this to get the
number of bytes used by a piddle if you like, but generally I do not recommend
using this as a setter in normal PDL code (that is, as a method for setting the
length of the piddle). The problems are as follows:

=over

=item fails for slices or mmap piddles

Manipulating a slice in the way that this function manipulates piddles will
cause Perl to crash with a segmentation fault if you call it on a slice. It will
break your mmapped piddle in less deadly but equally obnoxious ways if you
attempt to manipulate mmapped piddles. This routine will tell you the size of
any piddle, but it can only modify physical piddles and will croak if you
attempt to modify one of these types of piddles.

=item no data initialization

If you extend a piddle, this will simply allocate new memory for the piddle and
not initialize the new values. You will have garbage that you must clean up (or
overwrite with a memory transfer) before using the data in the piddle.

=item flattens piddles

This method will resize the piddle so it holds the desired number of bytes, but
it will create a one-dimensional piddle in the process. If you had multiple
dimensions, they will vanish! This may also cause problems with slices, as
discussed next.

=item action at a distance: may break slices

If you shorten a one-dimensional piddle, a slice of your piddle that points to
data that no longer exists will barf if you try to use it, saying:

 Slice cannot start or end above limit.

On the other hand, if you set the size of a multi-dimensional piddle using this
method, even if the new piddle is large enough to accomodate the old slice's
flattened array offsets, using that slice will barf with:

 Error in slice:Too many dims in slice.

=back

This function is provided for completeness, so that other code can call
L</SetSize> on an argument and it will "Do What I Mean," even for a piddle.
Just be aware that, under some circumstances, it will also do things that you
I<don't> mean, too.

Under the hood, this function uses the undocumented PDL method C<setdims> to
achieve its end. (I belive that method I<should> be documented; it's a C
function defined in pdlapi.c which is accessible as a Perl-level method. I
can't imagine that somebody went through the trouble of making that function
Perl-accessible without meaning that others use it. But clearly it has
side-effects that I was unable to tease out, leading to the trouble with slices
and mmapped piddles.)

=cut

sub PDL::nbytes {
	my $length = PDL::Core::howbig($_[0]->get_datatype) * $_[0]->nelem;
	return $length unless @_ > 1;
	
	my ($self, $new_length) = @_;
	
	croak("New size ($new_length) is not a multiple of this piddle's type, " . $self->type)
		unless $new_length % PDL::Core::howbig($self->get_datatype) == 0;
	
	# Warn if the underlying piddle has multiple dimensions:
	carp('Flattening multidimensional piddle with call to nbytes')
		if $self->ndims > 1;
	
	# Test if the underlying piddle is mmapped or a slice, in which case calling
	# setdims will cause major trouble:
	croak("Unable to call nbytes on a slice or a piddle with mmapped data")
		if _piddle_is_mmap_or_slice($self);
	
	# If we're here, we are good to go:
	$self->setdims([$new_length / PDL::Core::howbig($self->get_datatype)]);
	
	return $length;
}

1;
__END__

=head1 Unspecified launch failure

Normally CUDA's error status is reset to C<cudaSuccess> after calling
C<cudaGetLastError>, which happens when any of the functions in C<CUDA::Minimal>
croak, or when you manually call L</GetLastError>. With one exception, later
checks for CUDA errors should be ok unless B<they> actually had trouble. The
exception is the C<unspecified launch failure>, which places your CUDA context
in a non-recoverable state, and will cause all further memory allocations and
kernel launches to fail with C<unspecified launch failure>. You can still copy
memory to and from the device, but nontrivial kernel launches or memory
allocations should fail. With C<CUDA::Minimal>, the only way to recover from
this problem is to completely close the program.

The CUDA Toolkit for versions beyond 4.1 provides a function called
C<cudaDeviceReset>, which lets you reset the device without completely quitting
the program. Because CUDA::Minimal is meant to be a set of simple and incomplete
bindings, C<CUDA::Minimal> does not provide access to this function. If you find
that you need this function, you can write your own bindings using L<Inline::C> 
or incorporate such bindings into your own XS code. Note that calling
C<cudaDeviceReset> also invalidates your device pointers, so that you must copy
data off the device before resetting it. Put simply, recovery from a failed
kernel launch is very messy.

The best solution to this, in my opinion, is to make sure you have rock-solid
input validation before invoking kernels. If your kernels only know how to
process arrays that have lengths that are powers of 2, make sure to indicate
that in your documentation, and validate the length before actually invoking the
kernel. If your input to your kernels are good, this should not be a problem.

Note: There is a very confusing and tricky aspect to all of this. Tests in
F<t/z_kernel_invocations.t> seem to indiate that, at least with v5.5 of the
CUDA toolkit, trivial kernels that do not access memory will succeed after a
failed kernel launch! You can't do anything useful with that sort of kernel, of
course.

=head1 EXPORTS

This uses the standard L<Exporter> module, so it behaves in a fairly standard
way. I know it's considered modern to not export anything by default, but
I think that's dumb for this module. It exports all of the documented
functions. However, you can select to have only certain functions or groups of
functions exported if you wish.

=head2 Exporting nothing

If you're a purist and don't want anything in your current package, there's
nothing stopping you from saying

 use CUDA::Minimal '';

which won't import anything. If you only want one or two functions, you can
get at them with their fully-qualified name, like this:

 my $error_string = CUDA::Minimal::GetLastError;

or you can import specific functions by name:

 use CUDA::Minimal qw(GetLastError);

=head2 Memory functions

You can export the memory-related functions using the C<:memory> tag, like so:

 use CUDA::Minimal qw(:memory);

That will make L</Malloc>, L</MallocFrom>, L</Transfer>, and L</Free> available.

=head2 Synchronization

Suppose you have some module that invokes various CUDA stuff for you. It handles
all of the memory internally. If you want to benchmark that code, you will
want L</ThreadSynchronize>, which you get by either of the following:

 use CUDA::Minimal ':sync';
 use CUDA::Minimal 'ThreadSynchronize';

=head2 Error functions

If you want to have access to the error-handling functions L</GetLastError>,
L</PeekAtLastError>, and L</ThereAreCudaErrors>, you can use the C<:error> tag:

 use CUDA::Minimal qw(:error);

=head2 Utility functions

The functions for setting the size of host memory or determining the size of
an allocation are L</SetSize> and L</Sizeof>. You can import those using the
C<:util> tag:

 use CUDA::Minimal ':util';

=head1 TODO

=over

=item Proofread the documentation

For example, I'm not sure if all the warnings and error messages for the PDL
methods are documented at the moment.

=item Update the test suite

The functionality has ballooned since I wrote my original test suite. The suite
needs to be updated to include tests for the object-based interface, and to
check for throwing appropriate errors at appropriate times.

=item Objects to device pointers

The requirement for manual clean-up of device pointers could be done-away with
if L</Malloc> and L</MallocFrom> returned device pointer objects instead of
simple integers. In that case, the device memory would automatically clean 
itself up when it went out of scope, making calls to L</Free> almost entirely
unnecessary. (Or, for that matter, making it possible to free device memory
simply by saying C<$dev_memory = undef>.)

=back

=head1 BUGS AND LIMITATIONS

The code for CUDA::Minimal is hosted at github. Please file bugs at
L<https://github.com/run4flat/perl-CUDA-Minimal/issues>.

A potentially major shortcoming of this library is that it does not provide an
interface to global variables on the device. I have not decided if that's a bug
or a feature. I think it may be good because it requires kernel writers to
explicitly provide functions for manipulating global memory locations, which I
think is a good idea. Real-world usage will tell whether or not such a function
should be available.

The library also does not provide commands to a myriad of function calls
provided by CUDA-C (i.e. the 'Array' and 'Async' functions, not to mention
stream, event, and thread management, OpenGL, Direct3D, and many others). Since
Kirk and Hwu (see references below) never use those functions, I have never used
them, and see no need to include them in a minimal set of bindings. Those sorts
of bindings would be more approriate for a L<CUDA::Driver> library, which I hope
to take up some time soon, hopefully with your help! :-)

=head1 SEE ALSO

This module requires the nvcc tool chain and the Perl trappings for it,
which are discussed under L<ExtUtils::nvcc>.

An entirely different approach to CUDA that you can leverage from Perl is
L<KappaCUDA>.

I hope to some day write L<CUDA::Driver>, a complete wrapping of the CUDA
Driver API. However, not a single keystroke of code has been written (as of
this writing; as of your reading, things may have changed).

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

=cut
