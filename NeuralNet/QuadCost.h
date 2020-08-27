#pragma once


#include "Eigen\dense"


#ifdef PRECISE
using number = double;
#else
using number = float;
#endif


struct QuadCost {
	struct eval_der {
		number operator()(number activations, number outp) const {
			return activations - outp;
		}
	};
};
