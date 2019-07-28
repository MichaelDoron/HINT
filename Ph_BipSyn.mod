:----------------------------------------------------
:-metabolic glutamate/ionotropic glutamate
:-conductance increase as a function of glutamate
:-conductance with a temporal delay
:-no postsynaptic voltage dependence
:-alon.poleg-polsky@ucdenver.edu
:-2019
:----------------------------------------------------

NEURON {
POINT_PROCESS Ph_BipSyn
	RANGE BIPGhalf, BIPGslope
	RANGE BIPe,BIPg,BIPtau,BIPginf,glu,BIPgain
	RANGE PHg1,PHg2,PHtau1,PHtau2,Light
	NONSPECIFIC_CURRENT i
}
PARAMETER {
	:---bipolar cell
	BIPGslope=4		: the slope of glu dependence (0-no glu release, 1- full release)
	BIPGhalf=0.5		: the point of half glutamate release
	BIPgain=0.5			: the peak amplitude
	BIPe=-60			: reversal potential
	BIPtau=6			: rise/decay time
	:	for on (mGluR) BIPGslope=4, BIPe=-60,BIPtau=10
	:	for off (iglu) BIPGslope=4, BIPe=0,BIPtau=10
	:---photoreceptor
	PHtau1=5
	PHtau2=50
	:	for cone PHtau1=5,PHtau2=20
	:	for rod PHtau1=20,PHtau2=100
	Light=1			: is set by a vector play command from neuron
}

ASSIGNED {
	v (millivolt)
	i (nanoamp)
	BIPginf			: steady state conductance for a given glu level
	
	glu
}

STATE {
	BIPg				: conductance
	PHg1
	PHg2
}
 
BREAKPOINT {
	SOLVE state METHOD euler
	glu=0.5-(PHg1-PHg2)
	:if(glu<0){glu=0}
	:if(glu>1){glu=1}
	BIPginf= 1/(1+exp(-BIPGslope*(glu-BIPGhalf)))
	i = (1e-3)*BIPgain*(BIPg) *( (v - BIPe))
}
 
INITIAL {
	BIPg=0
	BIPginf=0
	PHg1=0
	PHg2=0
}

DERIVATIVE state {
	BIPg'=(BIPginf-BIPg)/BIPtau
	PHg1'=(Light-PHg1)/PHtau1
	PHg2'=(Light-PHg2)/PHtau2
}