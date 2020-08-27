#pragma once 
#include "Eigen\Dense"
#include "vector"
#include <numeric>
#include <algorithm>
#include <random>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <random>
#include <chrono>
#include <iostream>
#include <optional>

#include "mnist_loader.h"



using namespace Eigen;

#define PRECISE

#ifdef PRECISE
	using NeuralMat = MatrixXd;
	using NeuralVec = VectorXd;
	using number = double;
#else
	using NeuralMat = MatrixXf;
	using NeuralVec = VectorXf;
	using number = float;
#endif

template<typename ActivationFunction, typename CostFunction>
class NeuralNetwork {

	std::vector<NeuralMat> m_weights;  // weights of the network
	std::vector<NeuralVec> m_biases;  //  biases of the network
	std::vector<NeuralVec> m_zfcts;   //Should come up with a better name: this is the z function before the sigmoid is applied

	std::vector<NeuralVec> m_layerError;   //error at each layer
	std::vector<NeuralMat> m_derivweights;
	std::vector<NeuralVec> m_derivBias;

	std::size_t m_nLayers;
	std::size_t m_nHLayers;
	std::size_t m_batchSize;
	int m_nEpochs = 0;
	double m_learningRate;
	double m_regularization;
	std::optional<mnist_loader> m_validation_set;

public:
	std::vector<NeuralVec> m_activations; // contains the activation

	NeuralNetwork(const std::vector<int>&& sizes, int batchSize, int nEpochs, double learningRate, double regularization, std::optional<mnist_loader> valSet = std::nullopt) : m_learningRate{ learningRate } {
		// Initialize all dimensions 
		m_batchSize = batchSize;
		m_nEpochs = nEpochs;

		auto nLayers = sizes.size();
		m_nLayers = nLayers;
		m_nHLayers = nLayers - 1;
		m_activations.resize(nLayers);
		m_zfcts.resize(nLayers);
		m_layerError.resize(nLayers);

		std::random_device rd;
		std::mt19937 gen(rd());  
		std::uniform_real_distribution<number> dis(-1.0, 1.0);


		for (std::size_t i = 1; i < nLayers; ++i) {
			m_weights.emplace_back(NeuralMat::NullaryExpr(sizes[i], sizes[i - 1], [&]() {return dis(gen); }));
			m_biases.emplace_back(NeuralVec::NullaryExpr(sizes[i], [&]() {return dis(gen); }));
			m_derivweights.emplace_back(NeuralMat::Zero(sizes[i], sizes[i - 1]));
			m_derivBias.emplace_back(NeuralVec::Zero(sizes[i]));
		}

		for (std::size_t i = 0; i < nLayers - 1; ++i) {
			m_weights[i] /= sqrt(sizes[i]);
		}
		m_regularization = regularization;
		m_validation_set = valSet;

	};

	void sgd(std::vector<NeuralVec>& input, std::vector<NeuralVec> output) {
		std::vector<int> idx(input.size());

		for (int iidx = 0; iidx < input.size(); ++iidx) {
			idx[iidx] = iidx;
		}
		for (int iepoch = 0; iepoch < m_nEpochs; iepoch++) {
			std::cout << "Epoch: " << iepoch + 1 << "\n";
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
			for (std::size_t i = 0; i < input.size(); ++i) {
				backPropagation(input[idx[i]], output[idx[i]]);
				if (i % m_batchSize == 0) {
					for (int k = 0; k < m_derivBias.size(); k++) {
						m_derivBias[k] /=  m_batchSize;
						m_derivweights[k] /=  m_batchSize;
					}
					updateWeightsAndBiases();
					for (int k = 0; k < m_derivBias.size(); k++) {
						m_derivBias[k] *= 0.0;
						m_derivweights[k] *= 0.0;
					}
				}
			}
			if (m_validation_set) {
				auto accuracy = accuracyComputer(*m_validation_set, *this);
				std::cout << "Accuracy on validation data: " << accuracy << "\n";
			}
		}
	}

	NeuralVec feedForward(NeuralVec& input) {
		assert(input.size() == m_weights[0].cols());

		m_activations[0] = input;

		for (std::size_t i = 1; i < m_nLayers; ++i) {
			m_zfcts[i] = m_weights[i - 1] * m_activations[i - 1] + m_biases[i - 1];
			m_activations[i] = m_zfcts[i];
			m_activations[i] = m_activations[i].unaryExpr(&ActivationFunction::eval);
		}

		return m_activations[m_nLayers - 1];
	}

	int label() {
		auto outp = m_activations[m_nLayers - 1];
		for (int i = 0; i < outp.size(); ++i) {
			if (outp[i] > 0.5) {
				return i;
			};
		}
	}

private:

	void updateWeightsAndBiases() {
		for (std::size_t i = 1; i < m_nLayers; ++i) {
			auto regularizationPar = (1 - m_learningRate * (m_regularization));
			m_weights[i - 1] = regularizationPar*m_weights[i - 1] - m_learningRate * m_derivweights[i - 1];
			m_biases[i - 1] = m_biases[i - 1] - m_learningRate * m_derivBias[i - 1];
		}
	}

	void backPropagation(NeuralVec& input, NeuralVec& output) {
		auto activationsLastLayer = feedForward(input);
		auto lastLayerId = m_nLayers - 1;
		auto firstLayerId = 0;
		auto firstweightID = 0;
		auto lastweightID = m_nLayers - 2;
		auto derivAvctivationL = m_zfcts[lastLayerId];
		derivAvctivationL = derivAvctivationL.unaryExpr(&ActivationFunction::eval_der);
		NeuralVec errorLast = activationsLastLayer.binaryExpr(output, CostFunction::eval_der());
		errorLast = errorLast.cwiseProduct(derivAvctivationL);
		m_layerError[lastLayerId] = errorLast;
		m_derivBias[lastLayerId - 1] += errorLast;
		m_derivweights[lastLayerId - 1] += errorLast * m_activations[lastLayerId - 1].transpose();

		for (std::size_t currLayer = lastLayerId - 1; currLayer > firstLayerId; --currLayer) {  // Only hidden layers
			auto inner = (m_weights[currLayer].transpose() * m_layerError[currLayer + 1]);
			auto derActFunc = m_zfcts[currLayer].unaryExpr(&ActivationFunction::eval_der);
			m_layerError[currLayer] = inner.cwiseProduct(derActFunc);
			m_derivBias[currLayer - 1] += m_layerError[currLayer];
			m_derivweights[currLayer - 1] += m_layerError[currLayer] * m_activations[currLayer - 1].transpose();
		}
	}


};


template<typename ActivationFunction, typename CostFunction>
inline double accuracyComputer(mnist_loader& set, NeuralNetwork<ActivationFunction, CostFunction>& network) {
	int correctlyGuessed = 0;
	for (int i = 0; i < set.size(); ++i) {
		std::vector<double> image = set.images(i);
		std::vector<number> inp(image.begin(), image.end());
		NeuralVec nn = Map<NeuralVec, Eigen::Unaligned>(inp.data(), inp.size());
		NeuralVec out(10);
		out << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
		int label = set.labels(i);
		out[label] = 1;
		network.feedForward(nn);
		auto diff = network.label() - label;
		if (diff == 0) {
			correctlyGuessed++;
		}
	}

	return (double)correctlyGuessed / set.size();
}