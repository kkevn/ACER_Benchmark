# ACER_Benchmark

An easy to use web application for running common C/C++ and Cuda algorithms (currently two supported) on UIC ACER servers in the form of a benchmark test. The C/C++ and Cuda results can then be compared against one another in a visually meaningful way.

This project was done as a research project during my Summer 2019 internship for ACER@UIC. It was found that for parallelizable tasks, Cuda accelerated code is exponentially more efficient than traditional computations completed on the main processor.

*For more information, refer to the 'Usage' and 'How it Works' sections of the `/Documentation.pdf` found in this repository.*

---

## Run Instructions

*For detailed instructions, refer to the 'Installation/Setup' section of the `/Documentation.pdf` found in this repository.*

---

## Specifications

* **Java** for application logic
* **JavFX** for graphical user interface
* **JSch** for SSH and 2FA support
* **JPro** for browser support
* **Maven** for browser deployment
* **C++** for the benchmarks run on the processor
* **Cuda** for the benchmarks run on the graphics processor
