/* Single-Precision AX+Y in C/C++
 *******************************************************************
 * Description:
 * 	Populate two vectors each of size N. In the first vector, 
 * 	multiply each element by some constant scalar and then sum 
 * 	this product with with the elment at same index in other 
 * 	vector. This result gets stored in the second vector.
 *******************************************************************
 * Source:
 * 	https://devblogs.nvidia.com/even-easier-introduction-cuda/
 *******************************************************************
 */

#include <ctime>
#include <iostream>
#include <math.h>
#include <string>
using namespace std;

/* saxpy function */
void saxpy(int n, float a, float *x, float *y) {

	// iterate through all N elements
	for (int i = 0; i < n; i++)

	    // do AX+Y
	    y[i] = a * x[i] + y[i];
}

/* program's main() */
int main(int argc, char* argv[]) {

	// initialize default vector size and run count
	int N = 1000000;
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

	// print info
	cout << "========================================" << endl;
	cout << "|\tSingle-Precision AX+Y" << endl;
	cout << "========================================" << endl;
	cout << "|\tUsing C++11" << endl;
	cout << "|\tN = " << N << endl;
	cout << "|\tRuns = " << R << endl;
	cout << "|" << endl;
	cout << "|\trunning..." << endl;

	// initialize the float vectors
	float *x, *y;

	// allocate memory
	x = new float[N];
	y = new float[N];

	// populate vectors
	for (int i = 0; i < N; i++) {
		y[i] = 1.0f;
		x[i] = 2.0f;
	}

	// initialize clock
	clock_t start = clock();

	// perform algorithm R times
	for (int i = 0; i < R; i++) {

		// perform saxpy on CPU
		saxpy(N, 2.0f, x, y);
	}

	// stop clock
	clock_t stop = clock();

	// counter for errors (should be 5.0f = 2.0 x 2.0 + 1.0)
	int errors = 0;

	// iterate vector to check for errors
	for (int i = 0; i < N; i++) {

	    // increment error counter when unexpected value in index
	    if (fabs(y[i] - (4 * R + 1.0f)) > 0.0f)
		errors++;
	}

	// print end status
	cout << "|\t   done!" << endl;
	cout << "|" << endl;
	cout << "|\tCalculation Errors = " << errors << endl;
	cout << "|\tTime = " << (stop - start) / (double) CLOCKS_PER_SEC << " seconds" << endl;
	cout << "========================================" << endl;

	// free allocated memory
	delete[] x;
	delete[] y;
	
	return 0;
}

