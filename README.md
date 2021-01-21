# Blind Source Separation

## Heuristic Introduction

Consider a fixed setting of *N* sources and *M* receivers, denoted by s_i, x_i resp.

Assuming the mixed signal x_m at receiver *m* is a linear superposition of N unknown sources {s_n}

    x_m(t) = A_mn s_n(t)

where the instantaneous mixing matrix A_mn is fixed for all t by the spatial arrangement of the sources and the signal propagation.

For *N*=*M* the separation of the blind sources can be achieved by estimating and inverting the mixing matrix A. Denoting the inverted mixing matrix by W, the separated signals are given by

    s_n(t) = W_nm x_m(t)  for all t.

Now we find two ambiguities.

### Scaling ambiguity
By simultanously scaling the signals s_n by a factor *c* and the mixing matrix A_mn by a factor *1/c*, these factors cancel out and superposition does not change. But the magnitude of the separated signals will change.

(e.g. signal damping)

### Permutation ambiguity
The order of the sources is interchangeable, since the order of the terms in the superposition is commutative. 

(e.g. interchange speaker)

These ambiguities are demonstrated in the notebook **ambiguities_FastICA.ipynb**. 

# Frequency-Domain Blind Source Separation

Via STFT, we can approximate convolutive mixtures in the time domain as multiple instantaneous mixtures in the frequency domain. Thus, we can apply techniques from blind source separation for instantaneously mixed sources.
The signals in the frequency domain are described by complex numbers, so we need to employ a complex-valued ICA.

In the frequency domain, we need to solve the permutation ambiguity of an ICA solution. We need to group together the frequency components originating from the same source. The task is known as the permutation problem.
We focus on a method called time difference of arrival TDOA, followed by a clustering of frequency-dependent estimators.

## Convolutive Mixtures

The receivers are described by a M-dimensioal vector $$\vec x(t)$$, the sources by a N-dimensioal vector $$\vec (t)$$. The measured mixed signals are described by

$$
    \vec x(t) = \sum_{k=1}^N \sum_{l=0}^P \vec h_k(lt_s) s_k(t-lt_s)
$$

where $$\vec h_k$$ is the impulse response from source k

![Signal Overview](./images/setup.png)

## Short Time Fourier Transformation

The convolution factorizes in the frequency domain, so we have

$$
    \vec x(n,f) = \sum_{k=1}^N \vec h_k(f) s_k(n,f)
$$

where n is the frame index.

![Frequency Domain BSS](./images/fdbss.png)

## Separation

By calculating the separation matrix $$W(f)$$ via complex-valued ICA, we arrive at the separated frequency components $$y(n,f)$$

$$
    \vec y(n,f) = W(f)\vec x(n,f)
$$

## Ambiguities

For a non-singular separation, we can write

$$
    \vec x(n,f) = A(f)\vec y(n,f)
$$

where $$A=W^1$$. Neither the amplitude, nor the ordering of the separated signals 
is fixed by an ICA. The signal ordering can be changed by a permutation 
matrix $$P$$ with $$P^TP=I$$ and the scaling can be changed by a diagonal matrix $$\Lambda$$. 
Since we have $$I = P^T\Lambda^{-1}\LambdaP$$, the equation above is invariant 
under the following transformation

$$
\begin{aligned}
    & y \leftarrow \Lambda Py \\
    & A \leftarrow AP^t\Lambda^{-1}
\end{aligned}
$$

Finding a suitable transformations is the topic of the permutation problem and the scaling problem.
By clustering TDOA estimators we learn a permutation matrix for each frequency.

The scaling problem is fixed by bringing the time domain separated signal $$y_i(t)$$
close to the target source component $$x_{Ji}(t)$$ observed at receiver $$J$$.