:----------------------------------------------------
:-gap junction
:-alon.poleg-polsky@ucdenver.edu
:-2019
:----------------------------------------------------
NEURON {
	POINT_PROCESS gap
	NONSPECIFIC_CURRENT i
	RANGE i,preX,preY,postX,postY,gain
	POINTER vpre
}
PARAMETER {
	v (millivolt)
	vpre (millivolt)
	preX=0
	preY=0
	postX=0
	postY=0
	gain=0.3
}
ASSIGNED {
	i		(nanoamp)
}

BREAKPOINT {
	i=(v-vpre)*gain
}