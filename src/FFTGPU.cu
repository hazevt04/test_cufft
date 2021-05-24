
#include "FFTGPU.hpp"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_utils.hpp"


FFTGPU::FFTGPU( 
   const int new_num_samples, 
   const int new_fft_size, 
   const int new_seed,
   const mode_select_t new_mode_select,
   const std::string new_filename,
   const bool new_debug ):
      num_samples( new_num_samples ),
      seed( new_seed ),
      mode_select( new_mode_select ),
      filename( new_filename ),
      debug( new_debug ) {

   try {
      cudaError_t cerror = cudaSuccess;
      cufftResult cufft_result = CUFFT_SUCCESS;
      dout << __func__ << "(): num_samples is " << num_samples << "\n";

      int num_ffts = int(ceil(float(num_samples)/float(fft_size)));
      adjusted_num_samples = num_ffts * fft_size;

      dout << __func__ << "(): adjusted number of samples for allocation is " 
         << adjusted_num_samples << "\n";

      adjusted_num_sample_bytes = adjusted_num_samples * sizeof( cufftComplex );
      adjusted_num_frequency_bin_bytes = adjusted_num_samples * sizeof( cufftComplex );

      try_cuda_func_throw( cerror, cudaGetDevice( &device_id ) );
      
      cudaDeviceProp props;
      try_cuda_func_throw( cerror, cudaGetDeviceProperties(&props, device_id) );

      can_prefetch = props.concurrentManagedAccess;
      can_map_memory = props.canMapHostMemory;
      gpu_is_integrated = props.integrated;

      dout << __func__ << "(): can_prefetch is " << (can_prefetch ? "true" : "false") << "\n";
      dout << __func__ << "(): can_map_memory is " << (can_map_memory ? "true" : "false") << "\n";
      dout << __func__ << "(): gpu_is_integrated is " << (gpu_is_integrated ? "true" : "false") << "\n";

      stream_ptr = my_make_unique<cudaStream_t>();
      try_cudaStreamCreate( stream_ptr.get() );
      dout << __func__ << "(): after cudaStreamCreate()\n"; 
      
      samples.resize( adjusted_num_samples );
      frequency_bins.resize( adjusted_num_samples );

      samples.reserve( adjusted_num_samples );
      d_samples.reserve( adjusted_num_samples );
      frequency_bins.reserve( adjusted_num_samples );
      d_frequency_bins.reserve( adjusted_num_samples );

      char* user_env = getenv( "USER" );
      if ( user_env == nullptr ) {
         throw std::runtime_error( "Empty USER env. USER environment variable needed for paths to files" ); 
      }
      
      std::string filepath_prefix = "/home/" + std::string{user_env} + "/Sandbox/CUDA/test_cufft/";

      dout << __func__ << "(): filename is " << filename << "\n";
      dout << __func__ << "(): frequency_bin_filename is " << frequency_bin_filename << "\n";

      filepath = filepath_prefix + filename;
      frequency_bin_filepath = filepath_prefix + frequency_bin_filename;

      dout << __func__ << "(): filepath is " << filepath << "\n";
      dout << __func__ << "(): frequency_bin_filepath is " << frequency_bin_filepath << "\n"; 

      try_cuda_func_throw( cerror, cudaMemset( samples.data(), adjusted_num_sample_bytes, 0 ) );
      try_cuda_func_throw( cerror, cudaMemset( frequency_bins.data(), adjusted_num_frequency_bin_bytes, 0 ) );

      std::fill( samples.begin(), samples.end(), make_cuFloatComplex(0.f, 0.f) );
      std::fill( frequency_bins.begin(), frequency_bins.end(), make_cuFloatComplex(0.f, 0.f) );

      try_cufft_func_throw( cufft_result, cufftPlan1d( 
         &plan,
         fft_size,
         CUFFT_C2C,
         num_ffts ) );

   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   }
}


void FFTGPU::initialize_samples( ) {
   try {
      if( mode_select == mode_select_t::Sinusoidal ) {
         dout << __func__ << "(): Sinusoidal Sample Test Selected\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            float t_val_real = AMPLITUDE*cos(2*PI*FREQ*float(index)/float(num_samples));
            float t_val_imag = AMPLITUDE*sin(2*PI*FREQ*float(index)/float(num_samples));
            samples[index] = make_cuFloatComplex( t_val_real, t_val_imag );
         } 
      } else if ( mode_select == mode_select_t::Random ) {
         dout << __func__ << "(): Random Sample Test Selected\n";
         gen_cufftComplexes( samples.data(), num_samples, -50.0, 50.0 );
      } else if ( mode_select == mode_select_t::Filebased ) {
         dout << __func__ << "(): File-Based Sample Test Selected. File is " << filepath << "\n";
         read_binary_file<cufftComplex>( 
            samples,
            filepath.c_str(),
            num_samples, 
            debug );
      }           
      if (debug) {
         print_cufftComplexes( samples.data(), num_samples, "Samples: ",  " ",  "\n" ); 
      }
   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   } // end of try
} // end of initialize_samples( const FFTGPU::TestSelect_e test_select = Sinusoidal, 


void FFTGPU::gen_expected_results() {
   std::cout << "FFTGPU::" << __func__ << "(): Not yet implemented\n"; 
}


void FFTGPU::check_results( const std::string& prefix = "" ) {
   try {
      std::cout << "FFTGPU::" << __func__ << "(): " << prefix << "() Not yet ready\n";

      //float max_diff = 1;
      //bool all_close = false;
      //if ( debug ) {
      //   print_results( prefix + std::string{"Norms: "} );
      //   std::cout << "\n"; 
      //}
      //dout << __func__ << "():" << prefix << "frequency_bins Check:\n"; 
      //all_close = vals_are_close( frequency_bins.data(), exp_frequency_bins, num_samples, max_diff, "frequency_bins: ", debug );
      //if (!all_close) {
      //   throw std::runtime_error{ std::string{"Mismatch between actual frequency_bins from GPU and expected frequency_bins."} };
      //}
      //dout << "\n"; 
      
      //std::cout << prefix << "All " << num_samples << " Norm Values matched expected values. Test Passed.\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   }
}


void FFTGPU::print_results( const std::string& prefix = "Frequency Bins: " ) {
   print_cufftComplexes( frequency_bins.data(), num_samples, prefix.data(),  " ",  "\n" );
}


void FFTGPU::run() {
   try {
      cudaError_t cerror = cudaSuccess;
      cufftResult cufft_result = CUFFT_SUCCESS;

      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): adjusted_num_samples is " << adjusted_num_samples << "\n"; 
      
      initialize_samples();
      gen_expected_results();

      float gpu_milliseconds = 0.f;
      Time_Point start = Steady_Clock::now();
      
      //try_cuda_func_throw( cerror, cudaMemPrefetchAsync( samples.data(), adjusted_num_sample_bytes, device_id, nullptr ) );
      try_cuda_func_throw( cerror, cudaMemcpy( d_samples.data(), samples.data(), adjusted_num_sample_bytes, cudaMemcpyHostToDevice ) );

      // Run FFT
      try_cufft_func_throw( cufft_result, cufftExecC2C( plan, samples.data(), frequency_bins.data(), CUFFT_FORWARD ) );
      
      try_cuda_func_throw( cerror, cudaMemcpy( frequency_bins.data(), d_frequency_bins.data(), adjusted_num_sample_bytes, 
         cudaMemcpyDeviceToHost ) );

      //try_cuda_func_throw( cerror, cudaMemPrefetchAsync( frequency_bins.data(), adjusted_num_frequency_bin_bytes, cudaCpuDeviceId, 0 ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();

      check_results();

      std::cout << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";

      float samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 


   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   }
}


FFTGPU::~FFTGPU() {
   dout << __func__ << "() started\n";
   samples.clear();    
   frequency_bins.clear();

   if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );

   dout << __func__ << "() done\n";
}

