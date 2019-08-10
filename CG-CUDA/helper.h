#ifndef HELPER_H_
#define HELPER_H_

#define SIZE 2048
#define EPS 1e-14
#define MAX_ITER 1000

#define A(row,col) (A[(row)*SIZE + (col)])
#define b(x) (b[(x)])

float* generateA();
float* generateb();
void printMat(float* A);
void printVec(float* b);
float getMaxDiffSquared(float* a, float* b);
long getMicrotime();

#endif /* HELPER_H_ */
