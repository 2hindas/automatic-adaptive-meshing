# oresd-amr

This repository contains the code for an adaptive mesh refinement (amr) algorithm that can be applied to electromagnetic problems in geophysical applications. The algorithm iteratively refines an octree mesh which is used as a domain discretization by SimPEG to solve Maxwell's equations. The Operator Recovery Source Detector ([ORESD](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.969)) error estimator is used to find cells where to refine the octree mesh. The algorithm can be used on simple 3D models or a large geophysical model. 

`report.pdf` contains the technical details and experiments regarding the adaptive meshing algorithm.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. 

### Prerequisites

The repository runs on Python 3.7+ and the following packages are required for the code to run properly: 

```
SimPEG
discretize
numpy
matplotlib
scipy
```

### Getting the code

The code can not be installed, but is instead available by cloning this repository:

```
git clone git@github.com:emsig/oresd-amr.git
```

The files from the `src` folder can now be imported to run the meshing algorithm. Four examples of running the meshing algorithm are given in the `examples` folder. Three examples use a basic 3D geometric model and the fourth model uses a large geophysical model. 

## Authors

* Lars La Heij
* Tuhin Das
* Terrence Dahoe
* Rinto de Vries

## License

This project is licensed under the Apache V2 license. See `LICENSE.md` for more information.

## Acknowledgments

This project was made possible by and was supervised by 

* Prof. Dr. Ir. Evert Slob
* Dr. Dieter Werthmüller