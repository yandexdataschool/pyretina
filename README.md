# Artificial Retina algorithm via optimization
=======

This repo contains an implementation of the Artificial Retina algorithm for LHCb VELO tracking.

Currently contains:
- numerical optimization methods as main subroutine for local maximum search;
- multi-start as a global search method;
- 2 initial seeding algorithms;
- Retina response function as well as its gradient and Hessian matrix implemented via [theano](https://github.com/Theano/Theano) (thus supposed to be computed on a GPU or a multi-core CPU);
- simplified LHCb VELO simulation with parameters inspired by the upgrade TDR;
- utils for efficiency measuments.

Future work:
- primary vertex fitting;
- fine tuning of the meta-parameters (sigma cooling, optimizer's parameters);
- advanced initial seeding (based on hits and VELO's geometry);
- hit's timing;
- multi-stage helix curve fitting.
