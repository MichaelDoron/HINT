:----------------------------------------------------
:-metabolic glutamate/ionotropic glutamate
:-conductance increase as a function of glutamate
:-conductance with a temporal delay
:-no postsynaptic voltage dependence
:-alon.poleg-polsky@ucdenver.edu
:-2019
:----------------------------------------------------

NEURON {
POINT_PROCESS gradedSyn
	POINTER vpre
	RANGE Vhalf, Vslope
	RANGE e,g,tau,ginf,gain
	NONSPECIFIC_CURRENT i
}
PARAMETER {
	:---bipolar cell
	Vslope=1			: the slope of glu dependence (0-no glu release, 1- full release)
	Vhalf=-40			: the point of half glutamate release
	gain=0.5			: the peak amplitude
	e=0					: reversal potential
	tau=5				: rise/decay time
	:	for on (mGluR) BIPGslope=4, BIPe=-60,BIPtau=10
	:	for off (iglu) BIPGslope=4, BIPe=0,BIPtau=10
	vpre=-60			: is set by a vector play command from neuron
}

ASSIGNED {
	v (millivolt)
	i (nanoamp)
	ginf			: steady state conductance for a given presynaptic voltage level
}

STATE {
	g				: conductance
}
 
BREAKPOINT {
	SOLVE state METHOD euler
	ginf= 1/(1+exp(-Vslope*(vpre-Vhalf)))
	i = (1e-3)*gain*(g) *( (v - e))
}
 
INITIAL {
	g=0
	ginf=0
}

DERIVATIVE state {
	g'=(ginf-g)/tau
}