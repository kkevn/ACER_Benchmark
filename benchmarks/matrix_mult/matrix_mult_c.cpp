/* Matrix Multiplication AX+Y in C/C++
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
void matrix_mult(int n, int r, float *matrix) {

	// iterate through all n elements
	for (int i = 0; i < n; i++) {
	
	    // assign i / N at current index
	    matrix[i] = 1.0f * i / n;

	    // generate a new value at current index r times
       	    for (int j = 0; j < r; j++) {
		matrix[i] = matrix[i] * matrix[i] - 0.25f;
	    }
	}
}

/* program's main() */
int main(int argc, char* argv[]) {

	// initialize default array size and run count
	int N = 1024;
	int R = 1000;

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
		}
	}

	// calculate full size of matrix
	unsigned int N_squared = N * N;

	// print info
	cout << "========================================" << endl;
	cout << "|\tMatrix Multiplication" << endl;
	cout << "========================================" << endl;
	cout << "|\tUsing C++11" << endl;
	cout << "|\tN = " << N << "x" << N << " (=" << N_squared << ")"<< endl;
	cout << "|\tRuns = " << R << endl;
	cout << "|" << endl;
	cout << "|\trunning..." << endl;

	// initialize the float array (using 1D array to simulate 2D array or matrix)
	float *matrix;

	// allocate memory to matrix
	matrix = new float[N_squared];

	// initialize clock
	clock_t start = clock();
	
	// perform matrix mult on CPU
	matrix_mult(N_squared, R, matrix);

	// stop clock
	clock_t stop = clock();

	// print end status
	cout << "|\t   done!" << endl;
	cout << "|" << endl;
	cout << "|\tTime = " << (stop - start) / (double) CLOCKS_PER_SEC << " seconds" << endl;
	cout << "========================================" << endl;
	
	// free allocated memory
	delete[] matrix;

	return 0;
}

