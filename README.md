# pion_SSA
Code for computing the single spin asymmetry in elastic $\pi^0$ electroproduction

The expressions for the numerator of the SSA can be found in Eq. (72) of my notes. The denominator, given by the Primakoff contribution, can be found in Eq. (32) (after expanding in $\Delta$... I will add the exact expressions soon). 
The main code is `pionSSA.py`, which contains useful functions and the main class used for computing both the SSA and hPDFs. 

An example of how to calculate the SSA for different initial conditions is shown in `ssa_testing.ipynb`. An example of how to calculate the quark non-singlet hPDFs is shown in `pdf_testing.ipynb`. In the latter file, I reproduce the right side of Fig. (13) from 2308.07461.
