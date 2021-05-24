// C++ File for main

#include "FFTGPU.hpp"

#include "parse_args.hpp"

int main(int argc, char **argv) {
   try {
      my_args_t my_args;
      parse_args( my_args, argc, argv ); 

      if ( my_args.help_shown ) {
         return EXIT_SUCCESS;
      }

      FFTGPU fft_gpu{ my_args };

      fft_gpu.run();
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << "ERROR: " << __func__ << "(): " << ex.what() << "\n"; 
      return EXIT_FAILURE;

   }
}
// end of C++ file for main
