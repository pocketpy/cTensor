# cTensor
 
This is the prototype of a lightweight neural network library written in C11.

### GSoC 2025 Guide

https://pocketpy.dev/gsoc2025/ideas/#develop-math-operators-for-ctensor-library

This repo contains a `main.c` which is an example of how to use the library.
In this example, we are building a simple neural network with `linear` and `relu`.
This neural network is trained to learn from the "iris dataset".

The example is runable, but the result is not correct because the math operators have not been
implemented yet. If you are a student who is applying this project, please try to fix the math operators and make `main.c` work correctly.

#### How to Build and Run

Follow these steps to build and run the `cTensor` example. 

##### 1. Fork the Repository
Before cloning the repository, you need to fork it to your GitHub account. 
Go to the `cTensor` repository:  [https://github.com/pocketpy/cTensor](https://github.com/pocketpy/cTensor)   
In the top-right corner of the page, click the **Fork** button to create a copy of the repository under your GitHub account.

##### 2. Clone the Forked Repository
Now that you have forked the repository, clone it to your local machine using `git`:

```bash
git clone https://github.com/your-username/cTensor.git
```

##### 3. Install Dependencies
Once you've cloned the repository, navigate to the project directory:

```bash
cd cTensor
```

Ensure you have the necessary dependencies installed:

```bash
sudo apt-get install g++ 
sudo apt-get install cmake
```

##### 4. Generate the build files using CMake:

```bash
cmake .
```

##### 5. Build the project:

```bash
make
```

##### 6. Run the project:

```bash
./cten_exe
```
