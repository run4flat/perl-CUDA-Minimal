use Module::Build;

use strict;
use warnings;
use ExtUtils::nvcc;

use Module::Build;
my $class = Module::Build->subclass(
	class => 'My::Builder',
	code => q{
		sub compile_xs {
			my ($self, $file, %args) = @_;
			
			# Call the xs converter
			Module::Build::Base::compile_xs($self, $file, %args);
			
			# Fiddle with the generated C file a little
			print "cleaning troubling code in $file\n";
			
			# Get the c file name from the xs file name
			my $c_file = $file;
			$c_file =~ s/xs$/c/;
			
			# Back up the current c file so we can read it
			rename $c_file => "$c_file.backup";
			
			# Open handles to the input and output files
			open my $in_fh, '<', "$c_file.backup";
			open my $out_fh, '>', $c_file;
			
			# Copy everything except the lines involving PL_unitcheckav
			while(my $line = <$in_fh>) {
				next if $line =~ /PL_unitcheckav/;
				print $out_fh $line;
			}
			
			# All done; cleanup files
			close $in_fh;
			close $out_fh;
			unlink "$c_file.backup";
		}
	},
);

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
	config => {ExtUtils::nvcc::MB},
	include_dirs => '.',
);

#print "Compiler is ", $build->config('cc'), "\n";
#print "Linker is ", $build->config('ld'), "\n";

$build->create_build_script;

