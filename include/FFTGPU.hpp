#pragma once

#include "device_allocator.hpp"
//#include "managed_allocator_host.hpp"
//#include "managed_allocator_global.hpp"
//#include "man_vec_file_io_funcs.hpp"

#include "pinned_allocator.hpp"
#include "pinned_vec_file_io_funcs.hpp"

#include "my_args.hpp"
#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_utils.hpp"

#include <math.h>

constexpr float PI = 3.1415926535897238463f;
constexpr float FREQ = 1024.f;
constexpr float AMPLITUDE = 50.f;

class FFTGPU {
public:
   FFTGPU(){}
   
   FFTGPU( 
      const int new_num_samples, 
      const int new_fft_size, 
      const int new_seed,
      const mode_select_t new_mode_select,
      const std::string new_filename,
      const bool new_debug );

   FFTGPU( 
      my_args_t my_args ):
         FFTGPU(
            my_args.num_samples,
            my_args.fft_size,
            my_args.seed,
            my_args.mode_select,
            my_args.filename,
            my_args.debug ) {}
   

   void initialize_samples( );
   
   void gen_expected_results();

   void check_results( const std::string& prefix );

   void print_results( const std::string& prefix );
   
   void run();

   ~FFTGPU();
   
private:
 
   pinned_vector<cufftComplex> samples;
   pinned_vector<cufftComplex> frequency_bins;
   device_vector<cufftComplex> d_samples;
   device_vector<cufftComplex> d_frequency_bins;

   std::vector<cufftComplex> exp_frequency_bins;

   mode_select_t mode_select = default_mode_select;
   
   std::string filename = default_filename;
   std::string frequency_bin_filename = default_frequency_bin_filename;
   
   std::string filepath = "";
   std::string frequency_bin_filepath = "";

   int seed = default_seed;
   
   int fft_size = default_fft_size;
   int num_samples = default_num_samples;
   int adjusted_num_samples = default_adjusted_num_samples;
   bool debug = false;

   bool can_prefetch = false;
   bool can_map_memory = false;
   bool gpu_is_integrated = false;
   
   size_t num_sample_bytes = default_num_sample_bytes; 
   size_t adjusted_num_sample_bytes = default_adjusted_num_sample_bytes; 
   size_t num_frequency_bin_bytes = default_num_frequency_bin_bytes;
   size_t adjusted_num_frequency_bin_bytes = default_num_frequency_bin_bytes; 

   std::unique_ptr<cudaStream_t> stream_ptr;

   cufftHandle plan;
   int device_id = -1;

};


