use strict;
use warnings;
use File::Spec;
use Module::Build;

# Get nvcc's path

my $build = Module::Build->new(
	module_name => 'CUDA::Minimal',
	license => 'perl',
	build_requires => {
				'ExtUtils::CBuilder' => '0',
			},
	requires => {
				'perl' => '0',
			},
	dist_author => 'David Mertens <dcmertens.perl@gmail.com>',
	include_dirs => ['.', get_cuda_dir('include')],
	extra_linker_flags => [ '-L' . get_cuda_dir('lib'), '-lcudart'],
);

$build->create_build_script;



my $base_cuda_path;
sub get_cuda_dir {
	my $final_dir = shift;
	
	# Find the location of nvcc, and one directori up
	if (not defined $base_cuda_path) {
		my ($nvcc_dir) =	grep { -e $_ }
							map  { File::Spec->catfile($_,'nvcc') }
							File::Spec->path;
		die "Unable to find nvcc in your current path\n"
			unless defined $nvcc_dir;
		my ($volume, $directories) = File::Spec->splitpath($nvcc_dir);
		$directories = File::Spec->catdir($directories, '..');
		$base_cuda_path = File::Spec->catpath($volume, $directories, '');
	}
	
	return File::Spec->catdir($base_cuda_path, $final_dir);
}
