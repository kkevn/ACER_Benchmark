/* Matrix Multiplication AX+Y in Cuda
 *******************************************************************
 * Description:
 * 	Populate a float array of size N^2 with each index 
 * 	generated M times using mulitplication.
 *******************************************************************
 * Source:
 * 	https://stackoverflow.com/questions/7663343/simplest-possible-example-to-show-gpu-outperform-cpu-using-cuda
 *******************************************************************
 */

#include <ctime>
#include <iostream>
using namespace std;

/* matrix mult function */
__global__
void matrix_mult(int n, int r, float *matrix) {

	// give index to each thread
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// assign i / N at current index
	matrix[index] = 1.0f * index / n;

	// generate a new value at current index r times
       	for (int j = 0; j < r; j++) {
	    matrix[index] = matrix[index] * matrix[index] - 0.25f;
	}
}

/* program's main() */
int main(int argc, char* argv[]) {

	// initialize default array size and run count, and threads per block
	int N = 1024;
	int R = 1000;
	int T = 256;

	// assign new values to N (and R) if arguments provided
	if (argc > 2) {
		
		// iterate over arguments
		for (int i = 0; i < argc; i++) {
			
			// get current argument
			string arg = argv[i];

			// if size specified
			if (arg.compare("-n") == 0) {
                	        N = stoi(argv[i + 1]);
                	}

			// if run count specified
	                else if (arg.compare("-r") == 0) {
                        	R = stoi(argv[i + 1]);
        	        }

			// if thread count specified
	                else if (arg.compare("-t") == 0) {
                        	T = stoi(argv[i + 1]);
        	        }
		}
	}

	// calculate full size of matrix
	unsigned int N_squared = N * N;

	// print info
	cout << "========================================" << endl;
	cout << "|\tMatrix Multiplication" << endl;
	cout << "========================================" << endl;
	cout << "|\tUsing CUDA 9.2" << endl;
	cout << "|\tN = " << N << "x" << N << " (=" << N_squared << ")"<< endl;
	cout << "|\tRuns = " << R << endl;
	cout << "|\tThreads/Block = " << T << endl;
	cout << "|" << endl;
	cout << "|\trunning..." << endl;

	// initialize the float array (using 1D array to simulate 2D array or matrix)
	float *matrix;

	// allocate unified memory
	cudaMallocManaged(&matrix, N_squared * sizeof(float));

	// initialize clock
	clock_t start = clock();
	
	// perform matrix mult on CPU
	matrix_mult<<<(N_squared + T - 1) / T, T>>>(N_squared, R, matrix);

	// wait for GPU before continuing on CPU
	cudaDeviceSynchronize();

	// stop clock
	clock_t stop = clock();

	// print end status
	cout << "|\t   done!" << endl;
	cout << "|" << endl;
	cout << "|\tTime = " << (stop - start) / (double) CLOCKS_PER_SEC << " seconds" << endl;
	cout << "========================================" << endl;
	
	// free allocated memory
	cudaFree(matrix);

	return 0;
}

