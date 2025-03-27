# Shoal Of Fish

A visual simulation of boids (referred to as "fish" in this project). The presentation layer is implemented using the OpenGL interface. The simulation algorithm is provided in two equivalent approaches: CPU and GPU. Broadly speaking, the GPU approach is an optimization of the CPU approach, leveraging CUDA calls and the Thrust library.

![picture 1](https://raw.githubusercontent.com/StiffGitt/ShoalOfFishGPU/master/screens/pic1.png)

The first picture shows a simulation with 200 fish, demonstrating boid behavior.

![picture 2](https://raw.githubusercontent.com/StiffGitt/ShoalOfFishGPU/master/screens/pic2.png)

The second picture shows a simulation with 100,000 fish. Thanks to GPU optimization, the application maintains over 20 FPS in this scenario.

## Algorithm Overview

This simulation follows the standard approach to modeling large groups of boids (e.g., [Craig Reynolds' Boids](https://www.red3d.com/cwr/boids/)). Some rules have been tweaked to create more natural fish behavior, and additional features, such as a predator-prey mode, have been introduced to make the simulation more engaging.

A key optimization is the use of a spatial grid that divides the environment into smaller data structures, based on the fish positions. This significantly speeds up the algorithm by ensuring each fish only computes interactions with nearby boids.

## CUDA Optimization

The GPU version uses parallel CUDA and Thrust calls for all algorithmic steps, allowing smooth simulations even with tens or hundreds of thousands of objects. CUDA functions are primarily responsible for assigning fish to the correct grid cells and calculating each boid’s position and velocity, while the Thrust library helps efficiently manage operations on grid arrays.

## References

* [https://www.red3d.com/cwr/boids/](https://www.red3d.com/cwr/boids/)
* [https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html](https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html)
* [https://www.youtube.com/watch?v=PPsP1unDkSg](https://www.youtube.com/watch?v=PPsP1unDkSg)
