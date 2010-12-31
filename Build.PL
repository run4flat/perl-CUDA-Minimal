use Module::Build;
my $build = Module::Build->new(
	module_name => 'CUDA::Min',
	license => 'perl',
	build_requires => {
				'ExtUtils::nvcc' => '0.01',
#				'ExtUtils::CBuilder' => '0',
			},
	requires => {
				'perl' => '0',
			},
	dist_author => 'David Mertens <dcmertens.perl@gmail.com>',
	config => {cc => 'perl_nvcc', ld => 'perl_nvcc'},
	include_dirs => '.',
);

$build->create_build_script;

