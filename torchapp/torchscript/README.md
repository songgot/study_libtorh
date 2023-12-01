---
title: Torchscript model loading in C++
...

## Torchscript model loading in C++

There are two ways to use libtorch
1. After learning with pytorch, convert Weight to Torchscript and load it to libtorch.
2. Load as it is after learning with libtorch

Most of them use libtorch as number one. Because most open source is written in Python.

### Converting to TorchScript via tracing
Before proceeding with libtorch, we must first learn Python as mentioned above, and then proceed with trace conversion.

[convert_to_traceing.ipynb]
```
# convert_to_traceing.ipynb

# Convert your PyTorch model to TorchScript through tracing,
# An instance of the implemented model must be passed to the torch.jit.trace function along with example input values.
# Then, this function creates a torch.jit.ScriptModule object.
# The object created in this way will contain the results of the runtime trace when the model is executed in the module's forward method

import torch
import torchvision

# Create model instance
model = torchvision.models.resnet18()

# Input value typically passed to the model’s forward() method
example = torch.rand(1, 3, 224, 224)

# Create torch.jit.ScriptModule with tracing using torch.jit.trace
traced_script_module = torch.jit.trace(model, example)

# This traced ScriptModule can receive and process input values in the same way as a regular PyTorch module.
output = traced_script_module(torch.ones(1,3,224,224))
output[0, :5]

# Serializing a Script module to a file
# Once you have converted your model into a ScriptModule through tracing or annotating, you can now serialize it to a file.
# You can later read the module from the file using C++ and run it without any dependency on Python.
# For example, let's say we want to serialize the ResNet18 model we heard about in the tracing example.
# To serialize, just call the save function and pass the module and file name.
traced_script_module.save("./traced_resnet18.pt")

```
### Loading Script modules in C++
First, let's look at the code that loads the module. To load a serialized PyTorch model in C++, the application must use the PyTorch C++ API called LibTorch. LibTorch includes several shared libraries, header files, and CMake build configuration files. CMake is not a requirement for using LibTorch, but it is recommended and will continue to be supported in the future. First, let's look at the code that loads the module. We can easily load the module with the simple code below.

[main.cpp]
```
/* main.cpp */

/**
 * <torch/script.h> The header contains all the LibTorch libraries to run the examples.
 */
#include <torch/script.h> // Only one header file needed.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
	if (argc !=2) {
		std::cerr << "usage: torchscript <path-to-exported-script-module>\n";
		return -1;
	}

	torch::jit::script::Module module;
	try {
		/**
		 * Deserialize a ScriptModule from a file using torch::jit::load()
		 */
		module = torch::jit::load(argv[1]);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "ok\n";
	/**
	 * After successfully loading ResNet18 in C++, we can now run the module with just a few more lines of code. 
	 */

	/** 
	 * Create an input vector and add one input.
	 * To create an input tensor we use torch::ones(). This function is the C++ API version of torch.ones.
	 */
	std::vector<torch::jit::IValue> inputs, target;
	inputs.push_back(torch::ones({1, 3, 224, 224}));
	/**
	 * Now, if we pass the input vector to the forward method of script::Module and execute it, we will receive a new IValue returned.
	 * This value can be converted to a tensor via toTensor().
	 */
	at::Tensor output = module.forward(inputs).toTensor();
	/** Prints the first five values of the output value.*/
	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}

```
### Create CMakeLists.txt
```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(NAME torchscript)
project(${NAME})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${NAME} main.cpp)
target_link_libraries(${NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 17)
```
### How to build
For -DCMAKE_PREFIX_PATH, use the absolute path to libtorch.
```
sudo rm -rf build
cmake -B build -DCMAKE_INSTALL_PREFIX=/home/hyunil/libtorch
make -C build 
Or 
cd build; cmake --build . --config Release
```
### How to run
```
cd build
./torchscript ../traced_resnet18.pt
ok
-0.2440 -0.0810  0.5917 -0.1444  0.3061
[ CPUFloatType{1,5} ]
```
### Reference
https://tutorials.pytorch.kr/advanced/cpp_export.html
