# A collection of tests for the memory functions

use Test::More tests => 8;
use CUDA::Min ':all';
use strict;
use warnings;

# Make sure Malloc and MallocFrom croak in void context:
eval{ Malloc(10) };
like($@, qr/void context/, 'Malloc croaks in void context');
$@ = '';
eval{ MallocFrom(10) };
like($@, qr/void context/, 'MallocFrom croaks in void context');
$@ = '';

# Check that Malloc kicks when you do something stupid:
my $dev_ptr	= eval{ Malloc(-10) };
like($@, qr/Unable to allocate/, 'Malloc croaks when given stupid input');
$@ = '';

# Make sure that malloc works when asked for reasonable things:
$dev_ptr = eval{ Malloc(10) };
ok($@ eq '', 'Malloc works when asked for reasonable things');
$@ = '';

# Check basic behavior of Free
eval{ Free($dev_ptr) };
ok($@ eq '', 'Free works when given sensible input');
ok($dev_ptr == 0, 'Free sets scalars to zero after their memory is freed');

# Free should gripe when given stupid (or old) pointers:
eval{ Free(-10) };
like($@, qr/Unable to free memory on the device/, 'Free croaks when given stupid input');

# Finally, check that Free works with a list:
my $dev_ptr2 = eval{ Malloc(10) };
$dev_ptr = eval{ Malloc(10) };
$@ = '';
eval{ Free $dev_ptr, $dev_ptr2 };
ok($@ eq '', "Free behaves when given reasonable input");

