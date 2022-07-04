# Equilibrium


Basic Study
1. Weijie paper and activate the code: https://arxiv.org/pdf/2107.05533.pdf
2. Deep Equilibrium Models: http://implicit-layers-tutorial.org/deep_equilibrium_models/
3. Deep Equilibrium Paper: https://arxiv.org/pdf/2102.07944.pdf

K-Space: https://mriquestions.com/what-is-k-space.html

Deep learning for image registration: https://web.stanford.edu/~yplu/imgreg.pdf


### `Image Registration`

----
```diff
+ Definition: the process of transforming different sets of data into one coordinate system.

+ Purpose: Registration is necessary in order to be able to compare or integrate the data obtained from these different measurements.

- Be careful: 
```

#### [Image Registration] Medical Image Application
Medical image registration (for data of the same patient taken at different points in time such as change detection or tumor monitoring) often additionally involves elastic (also known as nonrigid) registration to cope with deformation of the subject (due to breathing, anatomical changes, and so forth). Nonrigid registration of medical images can also be used to register a patient's data to an anatomical atlas, such as the Talairach atlas for neuroimaging.

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177072414-72680097-75e6-4a74-b7b5-b6b3b178cc90.png">


<!--
Material: https://en.wikipedia.org/wiki/Image_registration
-->

----


## 


## Deep Equilibrium Models

### `[DEQ Model] 

`Equilibrium: a state of balance between opposing forces or actions`

<!--
Material: http://implicit-layers-tutorial.org/
-->

----
```diff
+ Goal 1: Express the entire deep network as an equilibrium computation

+ Goal 2: Find the fixed point directly via root finding rather than fixed point iteration alone.

- Be careful: 
```
----






## Deep Implicit Layers


Layer: Differentiable parametric function

Explicit vs. Implicit layers

Explicit layers: All commonly-used layers

Implicit layers examples: Differential equations, fixed point iteration, optimzation solutions

Explicit Layers  	      |      Implicit Layers
:---------------: | :-------------:
Computation graph for computing the forward pass, and backprop through it.  | Satisfying some joint condition of the input and output

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177073666-a100bf07-ad31-4c04-82d0-ab826a4a3fad.png">

Why use implicit layers?  	      
:---------------: 
Powerful representations: 
Memory efficiency
Simplicity:
Abstraction: 
