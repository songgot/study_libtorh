
/**
 * C++에서 Script 모듈 로딩하기
 * 직렬화된 PyTorch 모델을 C++에서 로드하기 위해서는, 어플리케이션이 반드시 LibTorch 라고 불리는 PyTorch C++ API를 사용해야합니다.
 * LibTorch는 여러 공유 라이브러리들, 헤더 파일들, 그리고 CMake 빌드 설정파일들을 포함하고 있습니다.
 * CMake는 LibTorch를 쓰기위한 필수 요구사항은 아니지만, 권장되는 방식이고 향후에도 계속 지원될 예정입니다.
 * 우선 모듈을 로드하는 코드에 대해 살펴보도록 하겠습니다. 아래의 간단한 코드로 모듈을 쉽게 읽어올 수 있습니다:
 */

/**
 * <torch/script.h> 헤더는 예시를 실행하기 위한 모든 LibTorch 라이브러리를 포함하고 있습니다.
 */
#include <torch/torch.h>
#include <torch/script.h> // 필요한 단 하나의 헤더파일.
// #include <torch/nn.h>
// #include <torch/optim.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

	auto x_data = torch::tensor({{0, 0}, {1, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 1}}, torch::kF32);
	auto y_data = torch::tensor({0, 1, 2, 0, 0, 2}, torch::kInt64);

	//std::cout << "x_data.size " << x_data.size(0) <<  std::endl;

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
		 * torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
		 */
		module = torch::jit::load(argv[1]);
	}
	catch (const c10::Error& e) {
		/**
		public c10::DistBackendError (Class DistBackendError)
		public c10::EnforceFiniteError (Class EnforceFiniteError)
		public c10::IndexError (Class IndexError)
		public c10::LinAlgError (Class LinAlgError)
		public c10::NotImplementedError (Class NotImplementedError)
		public c10::OnnxfiBackendSystemError (Class OnnxfiBackendSystemError)
		public c10::OutOfMemoryError (Class OutOfMemoryError)
		public c10::TypeError (Class TypeError)
		public c10::ValueError (Class ValueError)
		*/

		std::cerr << "error loading the model\n";
		return -1;
	}

	//std::cout << "ok\n";
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
			std::cout << "output:"<< output << std::endl;

			at::Tensor target = y_data.index({i}).data();
			//std::cout << "target:"<< target << std::endl;
			
			/** https://discuss.pytorch.org/t/c-loss-functions/27471 */
			/** dim = -1 => error : Dimension out of range (expected to be in range of [-1, 0], but got 1) */
			//F.cross_entropy()는 F.log_softmax()와 F.nll_loss()를 포함하고 있습니다.
			//F.cross_entropy는 비용 함수에 소프트맥스 함수까지 포함하고 있음을 기억하고 있어야 구현 시 혼동하지 않습니다.
			//여기서 nll이란 Negative Log Likelihood의 약자입니다. 위에서 nll_loss는 F.log_softmax()를 수행한 후에 남은 수식들을 수행합니다.
			auto loss = torch::nll_loss(torch::log_softmax(output, /*dim=*/-1), target); //손실값 계산 => 이상한 것 같음 
			//std::cout << "loss:"<< loss << std::endl;
			optimizer.zero_grad(); //기울기 초기화
			loss.backward(); //역전파 수행
			optimizer.step(); //가중치 업데이트
			if (i == x_data.size(0) -1) {
				std::cout <<"progress: "<< j <<" loss:"<< loss << std::endl;
			}
		}
	}

	module.save("model.pt");
}
