---
title: Load a pre-trained model and training again
...

## Load a pre-trained model and retrain model
First, please refer to [torchscript](https://github.com/songgot/study_libtorh/tree/master/torchapp/torchscript)

### Converting to TorchScript via tracing

[convert_to_traceing.ipynb]
```
# convert_to_traceing.ipynb

# Convert your PyTorch model to TorchScript through tracing,
# An instance of the implemented model must be passed to the torch.jit.trace function along with example input values.
# Then, this function creates a torch.jit.ScriptModule object.
# The object created in this way will contain the results of the runtime trace when the model is executed in the module's forward method

import torch
import torch.nn as nn

x_data = torch.Tensor([
    [0,0],
    [1,0],
    [1,1],
    [0,0],
    [0,0],
    [0,1]
])

y_data = torch.LongTensor([
    0, # etc
    1, # mammal
    2, # birds
    0,
    0,
    2
])

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.w1 = nn.Linear(2, 10)
        self.bias1 = torch.zeros([10])

        self.w2 = nn.Linear(10, 3)
        self.bias2 = torch.zeros([3])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y = self.w1(x) + self.bias1
        y = self.relu(y)

        y = self.w2(y) + self.bias2
        return y

# Create model instance
model = DNN()

# I think there is no need to declare (criterion and optimizer)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# It seems like we don’t have to train.
# for epoch in range(1000):
#    output = model(x_data)
#
#    loss = criterion(output, y_data)
#
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#
#    print("progress:", epoch, "loss=", loss.item())

# Input value typically passed to the model’s forward() method
example = torch.bernoulli(torch.full((6, 2), 0.5)) 

# Create torch.jit.ScriptModule with tracing using torch.jit.trace
traced_script_module = torch.jit.trace(model, example)

# This traced ScriptModule can receive and process input values in the same way as a regular PyTorch module.
output = traced_script_module(torch.ones(1,2))
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
/**
 * <torch/script.h> The header contains all the LibTorch libraries to run the examples.
 */
#include <torch/torch.h>
#include <torch/script.h> // Only one header file needed.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

	auto x_data = torch::tensor({{0, 0}, {1, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 1}}, torch::kF32);
	auto y_data = torch::tensor({0, 1, 2, 0, 0, 2}, torch::kInt64);

	std::cout << "x_data.size" << x_data.size(0) <<  std::endl;

	for (int i = 0; i < x_data.size(0); i++) {
		auto tensor_x = x_data.index({i});
		auto tensor_y = y_data.index({i});
		// std::cout << "Tensor at index " << i << ": " << tensor_x << std::endl;  
		// std::cout << "Tensor at index " << i << ": " << tensor_y << std::endl;
    }  

	if (argc !=2) {
		std::cerr << "usage: retrain_model <path-to-exported-script-module>\n";
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
	torch::jit::parameter_list params = module.parameters();
	std::vector<at::Tensor> paramsVector;

	/**ref: https://discuss.pytorch.org/t/adam-optimizer-with-jit-module/173175 */
	std::transform(params.begin(), params.end(), std::back_inserter(paramsVector),
                   [](const torch::jit::IValue& ivalue) { return ivalue.toTensor(); });
	torch::optim::SGD optimizer(paramsVector, torch::optim::SGDOptions(0.01));

	int progress = 0;
	int epochs = 1000;
	for (int j = 0; j < epochs; j++) {
		for (int i = 0; i < x_data.size(0); i++) {
			std::vector<torch::jit::IValue> input;
			//inputs.push_back(torch::ones({1, 2}));
			input.push_back(x_data.index({i}));

			at::Tensor output = module.forward(input).toTensor();
			//std::cout << "output:"<< output << std::endl;

			at::Tensor target = y_data.index({i}).data();
			//std::cout << "target:"<< target << std::endl;
			
			/** https://discuss.pytorch.org/t/c-loss-functions/27471 */
			/** dim = -1 => error : Dimension out of range (expected to be in range of [-1, 0], but got 1) */
			/* Calculate loss value */
			auto loss = torch::nll_loss(torch::log_softmax(output, /*dim=*/-1), target);
			//std::cout << "loss:"<< loss << std::endl;

			/* Gradient reset */
			optimizer.zero_grad();
			
			/* Perform backward */
			loss.backward(); 
			
			/* Weight update */
			optimizer.step(); 
			
			if (i == x_data.size(0) -1) {
				std::cout <<"progress: "<< j <<" loss:"<< loss << std::endl;
			}
		}
	}
	/** Save */
	module.save("model.pt");
}
```
### Create CMakeLists.txt
```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(NAME retrain_modeltorchscript)
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
./retrain_model ../pre_trained_model.pt

progress: 0 loss:0.191547
[ CPUFloatType{} ]
progress: 1 loss:0.189355
[ CPUFloatType{} ]
progress: 2 loss:0.186499
[ CPUFloatType{} ]
progress: 3 loss:0.184002
[ CPUFloatType{} ]
progress: 4 loss:0.18124
[ CPUFloatType{} ]
progress: 5 loss:0.179258
[ CPUFloatType{} ]
progress: 6 loss:0.176582
[ CPUFloatType{} ]
progress: 7 loss:0.174243
[ CPUFloatType{} ]
progress: 8 loss:0.172066
[ CPUFloatType{} ]
progress: 9 loss:0.169789
...

```

### classification by model.pt
Now, let’s classify using the saved model.pt file.
[classify.cpp]
```
#include <torch/torch.h>
#include <torch/script.h> // 필요한 단 하나의 헤더파일.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

	auto x_data = torch::tensor({{0, 0}, {1, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 1}}, torch::kF32);

	if (argc !=2) {
		std::cerr << "usage: classify <path-to-exported-script-module>\n";
		return -1;
	}

	torch::jit::script::Module module;
	try {
		/**
		 * torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
		 */
		module = torch::jit::load(argv[1]);
	}
	catch (const c10::Error& e) {

		std::cerr << "error loading the model\n";
		return -1;
	}
    for (int i = 0; i < x_data.size(0); i++) {
        std::vector<torch::jit::IValue> input;
        //inputs.push_back(torch::ones({1, 2}));
        input.push_back(x_data.index({i}));

        at::Tensor output = module.forward(input).toTensor();
        //std::cout << "output:"<< output << std::endl;
        auto max_val = output.flatten().max().item();  
        int max_idx = output.argmax().item().toInt();  
        std::cout << "output["<< max_idx <<"]: " << max_val << std::endl;
    }
}
```
Let's modify CMakeLists.txt
```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(NAME retrain_model)
project(${NAME})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${NAME} main.cpp)
add_executable(classify classify.cpp)

target_link_libraries(${NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(classify "${TORCH_LIBRARIES}")
set_property(TARGET classify PROPERTY CXX_STANDARD 17)
```
### How to run
```
cd build
./classify ./model.pt
output[0]: 4.25023
output[1]: 3.44921
output[2]: 5.38102
output[0]: 4.25023
output[0]: 4.25023
output[2]: 3.84064
```
The labels according to the input values are as follows.
```
x_data = torch.Tensor([
    [0,0],
    [1,0],
    [1,1],
    [0,0],
    [0,0],
    [0,1]
])
y_data = torch.LongTensor([
    0, # etc
    1, # mammal
    2, # birds
    0,
    0,
    2
])
```