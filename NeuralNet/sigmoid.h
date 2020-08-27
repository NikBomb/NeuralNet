#pragma once
#include <math.h>


#ifdef PRECISE
using number = double;
#else
using number = float;
#endif

struct Sigmoid {
	
	static number eval(number z) {
		return 1 / (1 + exp(-z));
	}

	static number eval_der(number z) {
		return eval(z) * (1 - eval(z));
	}
};
