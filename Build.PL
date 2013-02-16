use Module::Build;

use strict;
use warnings;
#use ExtUtils::nvcc;

my $class = Module::Build->subclass(
	class => 'My::Builder',
	code => q{
		sub new {
			my $class = shift;
			# Build the M::B object in the usual way:
			my $self = $class->SUPER::new(@_);
			 
			# Override some of the stuff important for PDL:
			$self->build_elements([qw(PL support cu pm xs share_dir pod script)]);
			
			$self->extra_linker_flags(
				@{$self->extra_linker_flags()},
				'-L/usr/local/cuda/lib', 
				'-lcudart',
			);
			
			return $self;
		}
		
		sub process_cu_files {
			my $self = shift;
			# Get all the .cu files in lib
			my $files = $self->rscan_dir('lib', qr/\.cu$/);
			# process each in turn
			$self->process_cu($_) foreach (@$files);
		}
		sub process_cu {
			# Based heavily on process_xs
			my ($self, $file) = @_;
			 
			# Build the object file name
			my $o_file = $file . 'o';
			
			# Build it with nvcc
			my @cmd = ('nvcc', '-o', $o_file, '-c', $file);
			print join(' ', @cmd), "\n";
			$self->do_system(@cmd);
			
			$self->extra_linker_flags(
				$o_file,
				@{$self->extra_linker_flags()},
			);
			
			$self->add_to_cleanup($o_file);
		}
	},
);


#my $build = Module::Build->new(
my $build = $class->new(
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
	#config => {ExtUtils::nvcc::MB('verbose')},
	include_dirs => '.',
);

#print "Compiler is ", $build->config('cc'), "\n";
#print "Linker is ", $build->config('ld'), "\n";

$build->create_build_script;

