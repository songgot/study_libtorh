
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
#include <torch/script.h> // 필요한 단 하나의 헤더파일.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
	if (argc !=2) {
		std::cerr << "usage: 001_example <path-to-exported-script-module>\n";
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

	std::cout << "ok\n";
	/**
	 * ResNet18 을 C++에서 성공적으로 로딩한 뒤, 이제 몇 줄의 코드만 더 추가하면 모듈을 실행할 수 있습니다. 
	 */

	/** 입력값 벡터를 생성하고 하나의 입력값을 추가합니다.
	 *  입력값 텐서를 만들기 위해서 우리는 torch::ones() 을 사용합니다. 이 함수는 torch.ones 의 C++ API 버전입니다.
	 */
	std::vector<torch::jit::IValue> inputs, target;
	inputs.push_back(torch::ones({1, 3, 224, 224}));
	/**
	 * 이제 script::Module 의 forward 메소드에 입력값 벡터를 넘겨주어 실행하면, 우리는 새로운 IValue 를 리턴받게되고,
	 * 이 값을 toTensor() 를 통해 텐서로 변환할 수 있습니다. 
	 */
	at::Tensor output = module.forward(inputs).toTensor();
	/** 출력값의 첫 다섯 값들을 프린트합니다.*/
	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
