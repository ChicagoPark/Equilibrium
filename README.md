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










## Deep Implicit Layers

----
##### `Outline`

(1) Backround and applications of implicit layers
(2) The mathematics of implicit layers
(3) Deep Equilibrium Models
(4) Neural ODEs
(5) Differentiable optimization
(6) Future directions

----

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





#### Motivating a simple implicit layer

Traditional deep network  	      |      Implicit Layers
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177086406-94c1d21a-4cfa-45f5-bed6-6050d91f7495.png">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177086397-e13db113-81dd-4270-aa82-8dba7d56c3b9.png">
|  | At every step, rather than adding a bias, we re-inject the input
|  | We use the same weight at each layer (weight-tied model)


#### Iterations of deep weight-tied models

----
```diff
@@ Background: As we repeat exact same function over again, we can view this system as dynamical system. @@

+ Key: We can design the network such that this iteration will converge to some fixed point, or equilibrium point.

# z: Hidden vector. It's acting like a state of the system
```
* `Recurrent Backpropagatino Network [(Minimal)Deep Equilibrium Model]`
> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177087570-91d8d20e-94f0-4d5f-9345-bf63c17b50f6.png">

* `Equilibrium point (fixed point)`
> <img width="100" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177087775-fbb49c7a-3acd-4203-9ccd-1c59ae186592.png">

----

Differentiation notation


































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
