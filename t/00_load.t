# Test that the library and the test library load:

use Test::More tests => 2;

use_ok('CUDA::Minimal');
use_ok('CUDA::Minimal::Tests');


