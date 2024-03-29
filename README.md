# Equilibrium


Basic Study
1. Weijie paper and activate the code: https://arxiv.org/pdf/2107.05533.pdf
2. Deep Equilibrium Models: http://implicit-layers-tutorial.org/deep_equilibrium_models/
3. Deep Equilibrium Paper: https://arxiv.org/pdf/2102.07944.pdf

K-Space: https://mriquestions.com/what-is-k-space.html

Deep learning for image registration: https://web.stanford.edu/~yplu/imgreg.pdf


### [Prerequisite] `Jacobian`

https://www.youtube.com/watch?v=wCZ1VEmVjVo


##### [Jacobian] Introduction
* Intuition: To calculate 2 inputs and 2 outputs functions.

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177341689-f6a9e1f7-16ea-4e9e-8b17-28cafbff5cc2.png">



##### `[Jacobian] Linear Maps`

----
```diff
+ Property 1: Parallel lines stay parallel
+ Property 2: All the spacings preserved with exact scaling factor
!             Determinant = scaling factor for areas or lengths
+ Property 3: Origin is fixed

- Be careful: 
```


> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177343487-551a3579-40cd-4b24-a062-1f435ad90098.png">
----



##### `[Jacobian] Derivatives in 1D`

----
```diff
+ Key 1: We want to know how it is different between `a` neighborhood and `f(a)` neighborhood.
+ Key 2: We consider the scale difference results from Linear Map.
+ Key 3: The value of Jacobian Matrix depends on `center of perspective (a)`
```

> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177348225-ce94a699-9f62-4952-8269-cca3a493bc15.png">
----


##### `[Jacobian] Derivatives in 2D`

----
```diff
+ Key 1: Jacobian Matrix depends on the center of perspective (a,b).
+ Key 2: Jacobian matrix is the matrix representing best linear map approximation of f near (a,b)
+ Key 3: Even 2D transformation doesn't look linear, when we zoom in, we can consider it as a Linear Map.         
+ Key 4: Jacobian determinant is how much areas scale near (a,b)

```
> <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177349789-5c13240c-ee27-4804-940a-983143e2b7b1.png">

----

##### `[Jacobian] Calculate Jacobian Matrix`

* We have function `f` and a point `(a,b)`.

* In Column 1, we fix the y value and take patial derivatives.

> <img width="400" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177352819-43bc8c3f-7bb8-4c0d-9862-2b7fdecb41b4.png">

* In Column 2, we fix the x value and take patial derivatives.

> <img width="400" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177353212-f0741964-1eb1-4e88-a7c8-993bcde13ac5.png">

> <img width="400" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177353373-c2acad7b-40ce-4640-a86b-d3f1117e02e4.png">

* Jacobian Matrix

> <img width="200" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177352315-15f2fad9-2352-48d0-a8e0-f3c749335f05.png">



### [Prerequisite] `Image Registration`

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


----
##### `Studylist`

(1) Non-linear root finding algorithm
(2) 
(3) 
(4) 
(5) 
(6) 

----







## Deep Implicit Layers

----
##### `Outline`

(1) Background and applications of implicit layers

(2) The mathematics of implicit layers

(3) Deep Equilibrium Models

(4) Neural ODEs

(5) Differentiable optimization

(6) Future directions

----

Layer: Differentiable parametric function



Explicit Layers  	      |      Implicit Layers
:---------------: | :-------------:
Computation graph for computing the forward pass, and backprop through it.  | Satisfying some joint condition of the input and output

> <img width="200" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177073666-a100bf07-ad31-4c04-82d0-ab826a4a3fad.png">

Simple-looking implicit equation can be (1) recurrent backprop models or deep equilibrium models; (2)differential equations(leading to Neural ODEs); (3) Optimality conditions of optimization problems(leadings to differentiable optimization approaches.)


Why use implicit layers?

> We can use the implicit function theorem to directly compute gradients at the solution point of these equations, without having to store any intermediate variables along the way.

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


  
## `Deep Equilibrium Models`

Equilibrium: a state of balance between opposing forces or actions`

----
```diff
+ Goal 1:     Express the entire deep network as an equilibrium computation

+ Goal 2:     Find the fixed point(z*) directly via root finding rather than fixed point iteration alone.

- Be careful: Find z* without performing the forward iteration, but by directly attempting to find a root of this equilibrium equation.
```


### `[DEQ Model] Application`

a variety of large-scale vision and NLP tasks


### `[DEQ Model] Deep networks and fixed point equations`

----
```diff
+ Key: Construct implicit layer and repeat this update an infinite times

! finding: When we iterate i by infinite time, for most “typical” deep layers the valued actually converge to a fixed point or equilibrium point: z*.

! Base story: We don't care to solve for the equilibrium point; but we can use any non-linear root finding algorithm to do so (and also to solve the backward pass).

+ Optimal formula: input injection Ux is required in the model. Because the equilibrium point doesn’t depend on any “initial” value of z1.
```

Since the output $(h(x))$ can be a different size as the hidden unit, we use a separate weight to produce the final output of the function

----

Traditional deep network  	      |      Implicit Layers
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177279685-f1a6e33b-96c8-45a9-8da0-0b674794207b.png">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177279692-1252ad71-82ac-4045-b7fb-883bbcb96bbf.png">
|  | $z_1 = 0$ <br /> $z_{i+1} = σ(Wz_i + Ux + b), i = 1, ..., k-1$ <br /> $ h(x) = W_kz_k + b_k$
|  | $z^* = σ(Wz^* + Ux + b)$


### `[DEQ Model] Properties of DEQs: Representational power, and implicit differentiation`

Let's find a fixed point of $z^* = σ(Wz^* + Ux + b)$


#### `Power of the DEQ representation`

----
```diff
+ Key 1: Any deep network - of any depth, with any connectivity - can be represented as a single layer DEQ model with the same number of parameters.

+ Key 2: We can concatenate all intermediate products of computation graph into the vector z, and have function f that applies the "next" computation.

! finding: 

+ Optimal formula: 

- Be careful: While this construction theoretically shows the power of a single DEQ layer, we should emphasize that this is not a construction that we actually use it practice.
```


##### `Proof of the DEQ flexibility`

(1) `Traditional composition of two functions`

$$y = g_2(g_1(x))$$

(2) Transfer traditional one into a single layer DEQ by `simply concatenating` all the intermediate terms of this function into a long vector.

$$f(z,x) = f(\begin{bmatrix}z_1 \\\z_2 \end{bmatrix},x) = \begin{bmatrix}g_1(x) \\\g_2(z_1) \end{bmatrix}$$


$$z^⋆=f(z^⋆,x)⟺z^⋆_1=g_1(x),z^⋆_2=g_2(z^⋆_1)=g_2(g_1(x))$$

> <img width="200" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177514164-92fe890b-ab4f-4076-8cd2-8ef2b1c685d7.png">

Note: if we compute an equilibirum point of this function, then the second component $z^⋆_2$ is precisely the output of the original concatenated network.


```diff
+ f: cell more than layer. Residual block, Transformer block, LSTM cell, etc
```


#### `DEQ Training`

##### `Forward pass`

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/178188889-1146e929-b3fd-41db-b451-c436d6680e94.png">


##### `Backward pass`

> <img width="350" alt="IMG" src="https://user-images.githubusercontent.com/73331241/178188898-176493f5-2ab6-4891-bb6d-e63096486674.png">

#### `DEQ Backpropagation`

----
```diff
+ Key 1: vector-jacobian product is the key aspect to integrating these DEQ layers within backpropagation

+ Key 2: vector-jacobian product helps us to integrate the DEQ layer within standard automatic differential tools.

+ Optimal formula: 

- Be careful: 
```
----

How DEQ can have relatively large number of parameter with small hidden unit?

> have the hidden layer "internal" to the residual cell be larger than the hidden unit exposed to the DEQ model.

#### Question

Which parameter will become equilibrium? The parameters of network?




How one layer is enough?
(1) Suppose we had a system that first computed an equilibrium of the function z⋆1=f1(z⋆1,x), then next computed an second equilibrium using z⋆1 as input, i.e., z⋆2=f(z⋆2,z⋆1).

(2) Again, however, it is possible to set this joint problem up as a single equilibrium problem instead, namely computing an equilibrium point of the system.

----

* Summary of DEQ approach

> <img width="400" alt="IMG" src="https://user-images.githubusercontent.com/73331241/177270482-64ffc1cb-0379-463b-ab19-ab3ad0a953a7.png">

----
