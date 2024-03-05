
# Shoal Of Fish
A visual simulation of boids (called fish in the project). The presentation layer is implemented using OpenGL interface. The simulation algorithm is implemented with two equivalent approaches, CPU and GPU. Broadly speaking the GPU project is optimalization of the CPU approach, using CUDA calls and Thrust library.

![picture 1](https://raw.githubusercontent.com/StiffGitt/ShoalOfFishGPU/master/screens/pic1.png)
In the first picture a simulation for 200 fish, presenting boid behaviour.

![picture 2](https://raw.githubusercontent.com/StiffGitt/ShoalOfFishGPU/master/screens/pic2.png)
In the second picture simulation for 100 000 fish, as seen above, through GPU optimization application runs in over 20 fps.
## Algorithm overview

The simulation algorithm is implementing the usual approach to modelling large group of boids (e.g. https://www.red3d.com/cwr/boids/). Although some rules are altered, so that fish behavior is more natural. Additionally I've added some extra rules, to make the simulation more intresting, like predator-prey mode. Finally a grid improvement, which groups fish in lesser data structures, based on their position in the grid spanning the entire board, which optimizes  algorithm by forcing fish to make the calculations only for boids that are close enough.
## CUDA optimization
The GPU version uses parallel CUDA or Thrust calls for all the algoritm steps making it possible to run fluent simulation for tens or hundreds of thousands of  objects. CUDA functions are mainly used for assigning fish to correct grid cells as well as calculating position and velocity for a particular boid. Thrust library is useful with grid array operations.

## References
* [https://www.red3d.com/cwr/boids/](https://www.red3d.com/cwr/boids/)
* https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html?fbclid=IwAR1OR63jNM5Ps8dC9-SXQLvabfEaNppvxrPvd3FUmVQt_v061esu1piTSAo
* https://www.youtube.com/watch?v=PPsP1unDkSg