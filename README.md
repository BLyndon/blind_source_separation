# Blind Source Separation

Consider a fixed setting of *N* sources and *M* receivers, denoted by s_i, x_i resp.

Assuming the mixed signal x_m at receiver *m* is linear superposition of N unknown sources {s_n}

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