#pragma once
#include <Windows.h>

/**
 * Perform a + b = c on element-wise operation.
 * \param a [in]
 * \param b [in]
 * \param n [in] array size of `a` and `b`. `a` and `b` must have the same size.
 * \param c [out] output array. the array size of c is n.
 */
void vectorAdd(const float *a, const float *b, float *c);

void vectorAdd(HANDLE a, HANDLE b, HANDLE c);

void runtest();
