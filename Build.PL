use Module::Build;

use strict;
use warnings;
use ExtUtils::nvcc;

use Module::Build;
my $build = Module::Build->new(
	module_name => 'CUDA::Minimal',
	license => 'perl',
	build_requires => {
				'ExtUtils::nvcc' => '0.03',
#				'ExtUtils::CBuilder' => '0',
			},
	requires => {
				'perl' => '0',
			},
	dist_author => 'David Mertens <dcmertens.perl@gmail.com>',
	config => {ExtUtils::nvcc::MB},
	include_dirs => '.',
);
$build->create_build_script;
