#include <iostream>
#include <vector>

#include "mnist_loader.h"
#include "NeuralNetwork.h"
#include "chrono"
#include "sigmoid.h"
#include "QuadCost.h"
#include "CrossEntropyFunc.h"

int main() {
	mnist_loader original_train("MNIST/train-images-idx3-ubyte",
		"MNIST/train-labels-idx1-ubyte");
	mnist_loader test("MNIST/t10k-images-idx3-ubyte",
		"MNIST/t10k-labels-idx1-ubyte");

	auto trainSize = original_train.size();
	int split = trainSize * 0.95;

	auto train = original_train.split(0, split);
	auto validation = original_train.split(split, trainSize);
	NeuralNetwork<Sigmoid, CrossEntr> n{ std::vector<int>{784,30,10}, 10, 30, 0.05, 0, validation };
	int maxEpochs = 10;
	int nEpochs = 0;

	
	std::cout << "Initial Accuracy: " << accuracyComputer(test, n) << "\n";

	std::vector<NeuralVec> inputs;
	std::vector<NeuralVec> outputs;

	// Prepare Data
	for (int i = 0; i < train.size(); ++i) {
		std::vector<double> image = train.images(i);
		std::vector<number> inp(image.begin(), image.end());
		inputs.emplace_back(Map<NeuralVec, Eigen::Unaligned>(inp.data(), inp.size()));
		NeuralVec out(10);
		out << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		int label = train.labels(i);
		out[label] = 1;
		outputs.emplace_back(out);
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	n.sgd(inputs, outputs);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	
	std::cout << "Final Accuracy: " << accuracyComputer(test,n) << "\n";
	std::cout << "Training Time = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << "[s]" << std::endl;

	return 0;
}