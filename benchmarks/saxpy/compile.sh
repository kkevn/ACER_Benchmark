# script to compile saxpy benchmarks

# compile benchmak based on parameter
case "$1" in
	# compile both
	"") g++ -std=c++11 saxpy_c.cpp -o c_cpp.out
	module load tools/cuda-9.2
	nvcc -std=c++11 saxpy_cuda.cu -o cuda.out;;

	# compile c/c++
        "cpp") g++ -std=c++11 saxpy_c.cpp -o c_cpp.out;;

	# compile cuda
        "cuda") module load tools/cuda-9.2
	nvcc -std=c++11 saxpy_cuda.cu -o cuda.out;;
esac
