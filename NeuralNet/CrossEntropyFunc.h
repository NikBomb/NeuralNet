#pragma once


#include "Eigen\dense"


#ifdef PRECISE
using number = double;
#else
using number = float;
#endif


struct CrossEntr {
	struct eval_der {
		number operator()(number a, number y) const {
			number guard = 1e-4;

			number a_ = a + guard;
			number one_a = 1.f - a + guard;
			
			return - y/a_ + ((1 - y) / one_a) ;
		}
	};
};