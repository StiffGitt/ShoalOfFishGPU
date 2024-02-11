#include "Fish.h"

void init_cuda(int n, int grid_length, Fish *fishes);
void free_cuda();
void make_calculations_cuda(float* vertices, float r1, float r2, float turnCoef, float cohensionCoef, float avoidCoef, float alignCoef, float predatorsCoef,
    float preyCoef, float maxV, float minV, float curX, float curY, float curActive, bool predatorMode);