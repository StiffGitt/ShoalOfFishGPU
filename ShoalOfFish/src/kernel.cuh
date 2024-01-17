#include "Fish.h"


void init_cuda(int grid_length);
void free_cuda();
void make_calculations_cuda(Fish* fishes, float r1, float r2, float turnCoef, float cohensionCoef, float avoidCoef, float alignCoef, float predatorsCoef,
    float preyCoef, float maxV, float minV, float curX, float curY, float curActive, bool predatorMode);