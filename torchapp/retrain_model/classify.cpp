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
