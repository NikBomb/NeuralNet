#pragma once

#include <vector>
#include <string>

class mnist_loader {
private:
	std::vector<std::vector<double>> m_images;
	std::vector<int> m_labels;
	int m_size;
	int m_rows = 0;
	int m_cols = 0;

	void load_images(std::string file, int num = 0);
	void load_labels(std::string file, int num = 0);
	int  to_int(char* p);
	mnist_loader(std::vector<std::vector<double>> images, std::vector<int> labels) : m_labels{ labels }, m_images{ images } {
		m_size = m_labels.size();
	};

public:
	mnist_loader(std::string image_file, std::string label_file, int num);
	mnist_loader(std::string image_file, std::string label_file);
	~mnist_loader();

	int size() { return m_size; }
	int rows() { return m_rows; }
	int cols() { return m_cols; }

	std::vector<double> images(int id) { return m_images[id]; }
	mnist_loader split(int start, int end);
	int labels(int id) { return m_labels[id]; }
};

