from nltk.corpus import stopwords
import nltk
from nltk import tokenize
from operator import itemgetter
import math

doc = ''' Introduction to Nonequilibrium Quantum Field
Theory
Jürgen Berges
Institut für Theoretische Physik, Universität Heidelberg
Philosophenweg 16, 69120 Heidelberg, Germany
http://www.thphys.uni-heidelberg.de/~berges
Abstract. There has been substantial progress in recent years in the quantitative understanding of
the nonequilibrium time evolution of quantum fields. Important topical applications, in particular in
high energy particle physics and cosmology, involve dynamics of quantum fields far away from the
ground state or thermal equilibrium. In these cases, standard approaches based on small deviations
from equilibrium, or on a sufficient homogeneity in time underlying kinetic descriptions, are not
applicable. A particular challenge is to connect the far-from-equilibrium dynamics at early times
with the approach to thermal equilibrium at late times. Understanding the “link” between the early-
and the late-time behavior of quantum fields is crucial for a wide range of phenomena. For the
first time questions such as the explosive particle production at the end of the inflationary universe,
including the subsequent process of thermalization, can be addressed in quantum field theory
from first principles. The progress in this field is based on efficient functional integral techniques,
so-called n-particle irreducible effective actions, for which powerful nonperturbative approximation
schemes are available. Here we give an introduction to these techniques and show how they can
be applied in practice. Though we focus on particle physics and cosmology applications, we
emphasize that these techniques can be equally applied to other nonequilibrium phenomena in
complex many body systems.
Based on summer school lectures presented at HADRON-RANP 2004, March 28 – April 3, 2004,
Rio de Janeiro, Brazil and at QUANTUM FIELDS IN AND OUT OF EQUILIBRIUM, September
23 – 27, 2003, Bielefeld, Germany.
Introduction to Nonequilibrium Quantum Field Theory
 1
CONTENTS
1Motivations and overview
1.1 How to describe nonequilibrium quantum fields from first principles? .
1.1.1 Standard approximation methods fail out of equilibrium . . . .
1.1.2 n-Particle irreducible expansions: universality and non-secularity3
4
6
10
2 n-Particle irreducible generating functionals I
2.1 Loop or coupling expansion of the 2PI effective action . . . . . . . .
 .
2.2 Renormalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 .
2.2.1 2PI renormalization scheme to order λR 2 . . . . . . . . . . . .
 .
2.2.2 Renormalized equations for the two- and four-point functions
 .
2.3 2PI effective action for fermions . . . . . . . . . . . . . . . . . . . .
 .
2.3.1 Chiral quark-meson model . . . . . . . . . . . . . . . . . . .
 .
2.4 Two-particle irreducible 1/N expansion . . . . . . . . . . . . . . . .
 .
2.4.1 Classification of diagrams . . . . . . . . . . . . . . . . . . .
 .
2.4.2 Symmetries and validity of Goldstone’s theorem . . . . . . .
 .
12
16
17
18
20
22
23
25
25
28
3Nonequilibrium3.13.23.33.43.53.63.7quantum field theory
Nonequilibrium generating functional . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
Initial conditions . . . . . . . . . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
Nonequilibrium 2PI effective action . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
Exact evolution equations . . . . . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
3.4.1 Spectral and statistical components . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
3.4.2 Detour: Thermal equilibrium . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
3.4.3 Nonequilibrium evolution equations . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
3.4.4 Non-zero field expectation value . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
3.4.5 Lorentz decomposition for fermion dynamics .
 .
 .
 .
 .
 .
 .
 .
 .
 .
Nonequilibrium dynamics from the 2PI loop expansion
 .
 .
 .
 .
 .
 .
 .
 .
 .
Nonequilibrium dynamics from the 2PI 1/N expansion
 .
 .
 .
 .
 .
 .
 .
 .
 .
3.6.1 Nonvanishing field expectation value . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
Numerical implementation . . . . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
 .
 .
29
29
31
33
34
35
37
38
42
43
47
49
50
51
4 Nonequilibrium phenomena
4.1 Scattering, off-shell and memory effects . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
4.1.1 LO fixed points . . . . . . . . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
4.1.2 NLO thermalization . . . . . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
4.1.3 Detour: Boltzmann equation . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
4.1.4 Characteristic time scales . . . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
4.2 Prethermalization . . . . . . . . . . . . . . . . . . . . . .
 .
 .
 .
 .
 .
 .
 .
4.3 Far-from-equilibrium field dynamics: Parametric resonance
 .
 .
 .
 .
 .
 .
 .
4.3.1 Parametric resonance in the O(N) model . . . . .
 .
 .
 .
 .
 .
 .
 .
55
57
57
61
65
68
71
75
77
5 Classical aspects of nonequilibrium quantum fields: Precision tests
5.1 Exact classical time-evolution equations . . . . . . . . . . . . . . . . .
5.2 Classicality condition . . . . . . . . . . . . . . . . . . . . . . . . . . .
88
89
92
Introduction to Nonequilibrium Quantum Field Theory
 2
5.3
Precision tests and the role of quantum corrections . . . . . . . . . . .
5.3.1 Classical equilibration and quantum thermal equilibrium . . . .
95
97
6 n-Particle irreducible generating functionals II: Equivalence hierarchy
 100
6.1 Higher effective actions . . . . . . . . . . . . . . . . . . . . . . . . .
 .
 102
6.1.1 4PI effective action up to four-loop order corrections . . . . .
 .
 104
6.1.2 Equivalence hierarchy for nPI effective actions . . . . . . . .
 .
 107
6.2 Nonabelian gauge theory with fermions . . . . . . . . . . . . . . . .
 .
 110
6.2.1 Effective action up to four-loop or O(g6 ) corrections . . . . .
 .
 112
6.2.2 Comparison with Schwinger-Dyson equations . . . . . . . . .
 .
 116
6.2.3 Nonequilibrium evolution equations . . . . . . . . . . . . . .
 .
 118
6.3 Kinetic theory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 .
 122
6.3.1 “On-shell” approximations . . . . . . . . . . . . . . . . . . .
 .
 122
6.3.2 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . .
 .
 127
78Acknowledgements
Bibliographical notes
128
129
1. MOTIVATIONS AND OVERVIEW
Cosmology is time evolution. Solid theoretical descriptions exist for the temporal and
spatial local thermal equilibrium related to the time evolution of a homogeneous and
isotropic metric and for small perturbations of this. Furthermore, there are powerful
numerical techniques for N-particle simulations. In contrast, research concerning the
interplay between fluctuations and the time evolution of fields is still scarce. Their
interaction is of crucial importance for the generation of density fluctuations during
the inflationary phase of the early universe and of entropy at the end of inflation. The
corresponding temperature fluctuations in the cosmic microwave background radiation
have led to spectacular high-precision measurements of cosmological parameters. It is
the dynamics of fluctuations which decides the question whether the baryon asymmetry
was created during a cosmological phase transition. Back reactions of large density
fluctuations on the evolution of the cosmic scale factor are also possible.
A frequently employed strategy is to concentrate on classical statistical field theory,
which can be simulated numerically. It gives important insights when the number of
field quanta per mode is sufficiently large such that quantum fluctuations are suppressed
compared to statistical fluctuations. However, classical Rayleigh-Jeans divergences and
the lack of genuine quantum effects — such as the approach to quantum thermal equi-
librium characterized by Bose-Einstein or Fermi-Dirac statistics — limit their use. A
coherent understanding of the time evolution in quantum field theory is required — a
program which has made substantial progress in recent years with the development of
powerful theoretical techniques. For the first time questions such as the explosive parti-
cle production at the end of the inflationary universe, including the subsequent process
of thermalization, can be addressed in quantum field theory from first principles. Ther-
malization leads to the loss of a substantial part of the information about the conditions
Introduction to Nonequilibrium Quantum Field Theory
 3
in the early universe. The precise understanding of phenomena out of equilibrium play
therefore a crucial role for our knowledge about the primordial universe. Important ex-
amples are the density fluctuations, nucleosynthesis or baryogenesis — the latter being
responsible for our own existence.
The abundance of experimental data on matter in extreme conditions from relativistic
heavy-ion collision experiments, as well as applications in astrophysics and cosmol-
ogy urge a quantitative understanding of far-from-equilibrium quantum field theory.
The initial stages of a heavy-ion collision require considering extreme nonequilibrium
dynamics. Connecting this far-from-equilibrium dynamics at early times with the ap-
proach to thermal equilibrium at late times is a challenge for theory. The experiments
seem to indicate early thermalization whereas the present theoretical understanding of
QCD suggests a much longer thermal equilibration time. To resolve these questions, it
is important to understand to what “degree” thermalization is required to explain the
observations. Different quantities effectively thermalize on different time scales and a
complete thermalization of all quantities may not be necessary. For instance, an approx-
imately time-independent equation of state p = p(ε ), characterized by an almost fixed
relation between pressure p and energy density ε , may form very early — even though
the system is still far from equilibrium! Such prethermalized quantities approximately
take on their final thermal values already at a time when the occupation numbers of indi-
vidual momentum modes still show strong deviations from the late-time Bose-Einstein
or Fermi-Dirac distribution. Prethermalization is a universal far-from-equilibrium phe-
nomenon which occurs on time scales dramatically shorter than the thermal equilibration
time. In order to establish such a behavior it is crucial to be able to compare between the
time scales of prethermalization and thermal equilibration. Approaches based on small
deviations from equilibrium, or on a sufficient homogeneity in time underlying kinetic
descriptions, are not applicable in this case to describe the required “link” between the
early and the late-time behavior.
A successful description of the dynamics of quantum fields away from equilibrium
is tightly related to the basic problem of how macroscopic irreversible behavior arises
from time-reversal invariant dynamics of quantum fields. This is a fundamental question
with most diverse applications. The basic field theoretical techniques, which are required
to understand the physics of heavy-ion collision experiments or dynamics in the early
universe, are equally relevant for instance for the description of the dynamics of Bose–
Einstein condensates in the laboratory.
1.1. How to describe nonequilibrium quantum fields from first
principles?
There are very few ingredients. Nonequilibrium dynamics requires the specification
of an initial state at some given time t0. This may include a density matrix ρD(t0) in
a mixed (TrρD 2 (t0) < 1) or pure state (TrρD 2 (t0 ) = 1). Nonequilibrium means that the
initial density matrix does not correspond to a thermal equilibrium density matrix:
ρD(t0) 6= ρD (eq)
 with for instance ρD (eq)
 ∼ e −β H for the case of a canonical thermal
ensemble with inverse temperature β . In contrast to close-to-equilibrium field theory,
Introduction to Nonequilibrium Quantum Field Theory
 4
the initial density matrix ρD (t0 ) may deviate substantially from thermal equilibrium.
This preempts the use of (non-)linear response theory, which is based on sufficiently
small deviations from equilibrium, or assumptions about the validity of a fluctuation-
dissipation relation. Since time-translation invariance is explicitly broken at initial times,
there will be no assumption about a sufficient homogeneity in time underlying effective
kinetic descriptions. In their range of applicability these properties should come out of
the calculation.
Completely equivalent to the specification of the initial density matrix ρD(t0)
is the knowledge of all initial correlation functions: the initial one-point function
Tr{ρ D(t0)Φ(t0, x)}, two-point function Tr{ρD(t 0)Φ(t0, x)Φ(t0 , y)}, three-point function
etc., where Φ(t, x) denotes a Heisenberg field operator. Typically, the “experimental
setup” requires only knowledge about a few lowest correlation functions at time t0 ,
whereas complicated higher correlation functions often build up at later times. The
question that nonequilibrium quantum field theory addresses concerns the behavior of
these correlation functions for t > t0 from which one can extract the time evolution of
other quantities such as occupation numbers. This is depicted schematically in the figure
below. Of course, one could equally evolve the density matrix in time and compute
observable quantities such as correlations from it at some later time. However, this
is in general much less efficient than directly expressing the dynamics in terms of
correlations, which we will do here (cf. Sec. 6).
Once the nonequilibrium initial state is specified, the time-evolution is completely
determined by the Hamiltonian or, equivalently, the dynamics can be described in terms
of a functional path integral with the classical action S. From the latter one obtains the
effective action Γ, which is the generating functional for all correlation functions of
the quantum theory (cf. Sec. 2). There are no further ingredients involved concerning
the dynamics than what is known from standard vacuum quantum field theory. Here we
consider closed systems without coupling to an external heat bath or external fields.
There is no course graining or averaging involved in the dynamics. In this respect, the
analysis is very different from irreversible phenomenological approaches. The fact that
the dynamics is obtained from an effective action automatically guarantees a number
of crucial properties for the time evolution. The most important one is conservation of
s
noitie.g. nonthermal
 Bose−Einstein/Fermi−Dirac
 d nos
 occupation number
 distribution
 cn n d i t i o n
 ?
 n
 t i i n a l i fo ocsl sai ot lin eI ω
 ω
 v i t ceffEt0
 early
 intermediate
 late
 time
Introduction to Nonequilibrium Quantum Field Theory
 5
energy. The analogue in classical mechanics is well known: if the equations of motion
can be derived from a given action then they will not admit any friction term without
further approximations.
It should be stressed that during the nonequilibrium time evolution there is no loss of
information in any strict sense. The important process of thermalization is a nontrivial
question in a calculation from first principles. Thermal equilibrium keeps no memory
about the time history except for the values of a few conserved charges. Equilibrium
is time-translation invariant and cannot be reached from a nonequilibrium evolution on
a fundamental level. It is striking that we will observe below that the evolution can
go very closely towards thermal equilibrium results without ever deviating from them
for accessible times. The observed effective loss of details about the initial conditions
can mimic very accurately the irreversible dynamics obtained from phenomenological
descriptions in their range of applicability.
1.1.1. Standard approximation methods fail out of equilibrium
For out-of-equilibrium calculations there are additional complications which do not
appear in vacuum or thermal equilibrium.1 The first new aspect concerns secularity:
The perturbative time evolution suffers from the presence of spurious, so-called secu-
lar terms, which grow with time and invalidate the expansion even in the presence of a
weak coupling. Here it is important to note that the very same problem appears as well
for nonperturbative approximation schemes such as standard 1/N expansions, where N
denotes the number of field components.2 Typically, secularity is a not a very difficult
problem and for a given approximation there can be various ways to resolve it. There
is a requirement, however, which poses very strong restrictions on the possible approxi-
mations: Universality, i.e. the insensitivity of the late-time behavior to the details of the
initial conditions. If thermal equilibrium is approached then the late-time result is uni-
versal in the sense that it becomes uniquely determined by the energy density and further
conserved charges. To implement the necessary nonlinear dynamics is demanding. Both
requirements of a non-secular and universal behavior can indeed be fulfilled using ef-
ficient functional integral techniques: so-called n-particle irreducible effective actions,
for which powerful nonperturbative approximation schemes are available. They provide
a practical means to describe far-from-equilibrium dynamics as well as thermalization
from first principles.
An illustrative example from classical mechanics. It is instructive to consider for a
moment the simple example of the time evolution of a classical anharmonic oscillator.
1
 This does not concern restrictions to sufficiently small deviations from thermal equilibrium, such as
described in terms of (non-)linear response theory, which only involve thermal equilibrium correlators in
real time.
2 Note that restrictions to mean-field type approximations such as leading-order large-N are insufficient.
They suffer from the presence of an infinite number of spurious conserved quantities, and are known
to fail to describe thermalization. Secularity enters the required next-to-leading order corrections and
beyond. (Cf. Sec. 4.)
Introduction to Nonequilibrium Quantum Field Theory
 6
This exercise will explain some general problems, which one encounters using perturba-
tive techniques for initial-value problems, and indicates how to resolve them. Below we
will consider an illustrative “translation” of the outcome to the case of nonequilibrium
quantum fields. One of the benefits will be that important concepts such as n-particle
irreducible effective actions appear here in a very intuitive way, before they will be thor-
oughly discussed in later sections.
Our damped oscillator with time-dependent amplitude y(t) is characterized by an
infinite number of anharmonic terms:
ÿ + y = −ε ẏ − (ε y)3 − (ε y)5 − (ε y)7 − . . .
 ,
 ε ≪ 1
 (1)
Here each dot denotes a derivative with respect to time t. The presence of anharmonic
terms to arbitrarily high order is reminiscent of the situation in quantum field theory
(QFT), where the presence of quantum fluctuations can induce self-interactions to high
powers in the field. Consider the ideal case for perturbative estimates, i.e. the presence
of a small parameter ε which suppresses the contributions from higher powers in y.
The example is chosen such that it can be easily solved without further approximations
numerically by summing the geometric series with ∑∞
 n=3 (ε y)n
 = (ε y)3
 /(1 − ε 2 y2
 ). For
the above second-order differential equation the intial-value problem is defined by giving
the amplitude and its first derivative at initial time t = 0. We employ e.g. y(0) = 1,
ẏ(0) = −ε /2 and consider the time evolution for t ≥ 0. One finds the expected damped
oscillator behavior displayed below for ε = 0.1:
y
1
0.75
0.5
0.25
t
20
 40
 60
 80
-0.25
-0.5
-0.75
The question is whether an accurate approximate description of the full y(t) is possible
if contributions from higher powers of ε are neglected? In view of the presence of the
small expansion parameter this is certainly possible, however, standard perturbation
theory fails dramatically to provide a good description. A perturbative expansion of y
in ε ,
ε 2
ypert(t) = y 0 (t) + ε y1 (t) +
 y2 (t) + O(ε 3) ,
 (2)
2
yields the standard hierarchy of equations:
ÿ0 (t) + y 0(t) = 0
 ,
 ÿ1 (t) + y1(t) = −ẏ0
 ,
 ...
 (3)
These can be iteratively solved as
1
 1
y0 (t) = 2
 eit + c.c.
 ,
 y1 (t) = − 4
 eit t + c.c.
 ,
 ...
 (4)
Introduction to Nonequilibrium Quantum Field Theory
 7
Doing this to second order in ε one finds:
ypert (t) = 1 2
 eit
 
1 − 2
 ε
 t +
 ε 8
 2  t 2
 − it 
 
 + c.c.
 (5)
Beyond the lowest order, the solution contains secular terms which grow with powers
of the time t and one arrives at the important conclusion:
The perturbative expansion is only valid for ε t ≪ 1
We now consider an alternative (“nPI type”) expansion: For this we classify the terms
of the equation of motion (1) according to powers of the small parameter ε . The lowest
order takes into account all terms of the equation to order ε 0 , which to this order gives
the same as in perturbation theory:
(0)
 (0)
 (0)
 1
ÿnPI + ynPI = 0
 ⇒
 ynPI (t) = 2
 eit + c.c.
 (6)
The next order takes into account all terms of (1) to order ε :
(2)
 (2)
 (2)
 (2)
 1 √1−ε 2
/4−εt/2ÿnPI + ynPI = −ε ẏnPI ⇒ ynPI (t) = 2
 eit + c.c.
 (7)
Since there are no ε 2 terms appearing in (1), this also agrees with the second order of
the expansion. The next order would take into account all terms of the equation to order
ε 3 and so on. This provides a systematic approximation procedure in terms of powers
of the small parameter ε . However, in contrast to the perturbative expansion the “nPI
type” expansion turns out to be non-secular in time. One explicitly observes that both
the lowest order approximation and ynPI (2)
 (t) do not exceed O(ε 0 ) for all times. Therefore,
the result (7) provides an approximation to the equation (1) up to quantitative corrections
of order ε 3 at all times. The already very good agreement with the exact result can be
checked explicitly.
It is instructive to compare the “nPI type” solution to the perturbative one by expand-
ing (7) in powers of ε :
(2)
 1 it √1−ε 2 /4−ε t/2
ynPI (t) =
 e
 + c.c.
2(!)
 =
 2
 1 eit
 
1 − 2
 ε
 t +
 ε 8
 2 t 2
 − it 

 + O(ε 3 ) + c.c.
= ypert (t) + O(ε 3) + c.c.
We conclude that
•
 the “nPI” result corresponds to the perturbative one up to the order of approxima-
tion (here up to O(ε 3 ) corrections).
• the “nPI” result resums all secular terms to infinite perturbative order in ε .
Introduction to Nonequilibrium Quantum Field Theory
 8
Stated differently: infinite summation of perturbative orders is required to obtain a
uniform approximation to the exact solution, i.e. that the error stays of a given order
at all times.
What was the reason for the “success” of this alternative expansion? It is based on a
standard procedure for differential equations in order to enlarge the convergence radius
of an expansion. In the differential equation one neglects contributions from higher
powers of the expansion parameter. However, in contrast to perturbation theory one
does not expand in addition the variable y. This procedure is sometimes called “self-
consistent” since at each order in the approximation the dynamics is solely expressed
in terms of the dynamical degree of freedom itself. For instance, in Eq. (7) both on the
left and on the right hand side of the differential equation appear derivatives of the same
(2)
variable y . In contrast, in the perturbative hierarchy (3) the presence of “external”
nPIoscillating “source terms” such as y0 driving y1 leads to secular behavior. The procedure
of the “nPI type” expansion is simple and turns out to be very efficient to achieve
non-secular behavior: For given dynamical degrees of freedom truncate the dynamics
according to powers of a small expansion parameter. Of course, the choice of the degrees
of freedom is always based on physics. The price to be paid for this expansion scheme
is that at some order one necessarily has to solve nonlinear equations without further
approximations. However, as we will see next, it is precisely the nonlinearity which is
required to be able to obtain universality in the sense mentioned above.
In order to illustrate properly the aspect of universality the single oscillator exam-
ple (1) is too simple. In order to include more degrees of freedom we add a “three-
momentum” label to the variables and consider:
ẏp ∼
 Z
qk
 
(1 + yp)(1 + yq )ykyp−q−k − ypyq (1 + yk )(1 + yp−q−k)
 .
 (8)
Here the integrals on the r.h.s. involve momenta q and k above some suitable value, and
one observes that the time derivative of yp is proportional to a “gain” and a “loss” term.
This is like in a Boltzmann equation, which describes the rate of scattering of particles
into the state with momentum p minus the rate of scattering out of that momentum
(cf. Sec. 4.1.3). If the nonlinear equation (8) is solved without further approximations
then the late-time solution is given by the well-known result
1
yp →
 eβ (|p|− μ ) 1
 ,
 (9)
−with real parameters β and μ , which for the Boltzmann equation are identified as tem-
perature and chemical potential respectively. The form of the solution (9) is universal,
i.e. completely independent of the details of the initial condition for the solution of (8).
We emphasize that the Boltzmann type equation (8) is “self-consistent” in the sense
mentioned above. In contrast, a non–“self-consistent” approximation will not show the
desired universality in general. For instance, consider (8) in a linearized approximation
ẏp = (1 + yp ) σp 0 − yp σ 0 p ,
 (10)
Introduction to Nonequilibrium Quantum Field Theory
 9
Eq. with (10) time-independent has the solution
 σp 0 ∼
 R
qk(1 + yq (0))yk(0)yp−q−k (0)
 and equivalently for σ p 0 .
yp = σp γp
 0 0
 + "
 yp (0) − σp γp
 0 0 #
 e−γp 0t
 ,
 (11)
with γp 0 0 = σp 0 − σ 0 p . For late times γp 0 t ≫ 1 this always depends on the chosen initial
σp 0/γp and is, of course, only useful if they are chosen to be already the equilibrium
values.
1.1.2. n-Particle irreducible expansions: universality and non-secularity
One should not misunderstand the following illustrative “translation” of the above
mechanics examples to QFT. It is intended to give intuitive insight. The questions of
secularity and universality are of course settled by actual nonequilibrium calculations in
QFT, which is the topic of the main body of this text. For a moment, let us transcribe the
above results to the language of QFT. The amplitude y of the above oscillator example
plays the role of the one-point function, i.e. the field expectation value or macroscopic
field. The amplitude squared plays the role of the two-point function or propagator, the
cubic amplitude that of the three-point function or proper three-vertex, etc.:
y(t)
y2 (t)
y3 (t)
y4 (t)
..
.
” → ”
” → ”
” → ”
” → ”
φ (x) = hΦ̂(x)i
(macroscopic field)
G2 (x, y) = hT Φ̂(x)Φ̂(y)i − φ (x)φ (y) ≡ G(x, y)
(propagator)
G3 (x, y, z) or, with G3 ∼ GGGV 3 :
V 3 (x, y, z)
 (proper three-vertex)
V 4 (x, y, z, w)
 (proper 4-vertex)
where the symbol T denotes time-ordering. The knowledge of all n-point functions
provides a full description of the quantum theory. In contrast to the classical example,
the information contained e.g. in the two-point function cannot be recovered from the
one-point function: G 6∼ φ 2 etc. In QFT all n-point functions φ , G, V 3, V 4 . . . constitute
the set of dynamical “degrees of freedom”.
The equations of motions for all n-point functions are conveniently encoded in a
generating functional or effective action. There are different functional representa-
tions of the effective action. The so-called n-particle irreducible (nPI) effective action
Γ[φ , G,V 3, . . . ,V n] is expressed in terms of φ , G, V 3 , . . . , V n and is particularly efficient for
a description of suitable approximation schemes in nonequilibrium quantum field the-
ory. It determines the equations of motion for φ , G, V 3, . . . , V n by first-order functional
Introduction to Nonequilibrium Quantum Field Theory
 10
derivatives or so-called stationarity conditions:
δ Γ[φ , G,V 3 , . . .]
 δ Γ[ φ , G,V 3 , . . .]
 δ Γ[φ , G,V3 , . . .]
= 0 ,
 = 0 ,
 = 0 , ...
δφ
 δ G
 δ V 3
(12)
The above “nPI” approximation scheme for the oscillator example reads now:
•
 Classify the contributions to Γ[φ , G,V 3, . . .] according to powers of a suitable ex-
pansion parameter (e.g. coupling/loops, or 1/N). In terms of the “variables” φ , G,
V 3 . . . there is a definite answer for the approximate effective action to a given order
of the expansion.
• The equations of motion for φ , G, V 3 . . . are then determined from the approximate
effective action according to (12) without any further assumptions.
In general this leads to nonlinear integro-differential equations, which typically require
numerical solution techniques if one wants to provide the “link” between the early and
the late-time behavior. We emphasize that nonlinearity is a crucial ingredient for late-
time universality. The reward is that one can compute far-from-equilibrium dynamics as
well as the subsequent approach to thermal equilibrium in quantum field theory from first
principles (cf. Sec. 4). In turn this can be used to derive powerful effective descriptions
and to determine their range of applicability.
In order to be useful in practice it is crucial to observe that it is not necessary to
calculate the most general Γ[φ , G,V 3 , . . . ,V n ] for arbitrarily large n. There exists an
equivalence hierarchy between nPI effective actions. For instance, in the context of a
loop expansion one has:
Γ(1loop) [ φ ]Γ(2loop) [ φ ]Γ(3loop) [ φ ]..
.
=6=6=Γ(1loop) [φ , D] = . . . ,
Γ(2loop) [φ , D] = Γ(2loop) [ φ , D,V 3 ] = . . . ,
Γ(3loop) [φ , D] 6= Γ(3loop) [ φ , D,V 3 ] = Γ(3loop) [φ , D,V 3 ,V 4] = . . . ,
where Γ(n−loop) denotes the approximation of the respective effective action to n-th
loop order in the absence of external sources. E.g. for a two-loop approximation all
nPI descriptions with n ≥ 2 are equivalent and the 2PI effective action captures already
the complete answer for the self-consistent description up to this order. In contrast, a
self-consistently complete result to three-loop order requires at least the 3PI effective
action in general, etc. There are, however, other simplifications which can decrease
the hierarchy even further such that lower nPI effective actions are often sufficient in
practice. For instance, for a vanishing macroscopic field one has Γ2PI (3loop)
 [φ = 0, G] =
Γ4PI (3loop)
 [φ = 0, G,V 3 = 0,V 4 ] = . . . in the absence of sources. Typically the 2PI, 3PI
or maybe the 4PI effective action captures already the complete answer for the self-
consistent description to the desired/computationally feasible order of approximation.
Introduction to Nonequilibrium Quantum Field Theory
 11
2. N-PARTICLE IRREDUCIBLE GENERATING FUNCTIONALS I
In this section we discuss the construction of the one-particle irreducible (1PI) and the
two-particle irreducible (2PI) effective action. The latter provides a powerful starting
point for systematic approximations in nonequilibrium quantum field theory. To be
specific, we consider first a quantum field theory for a real, N–component scalar field φa
(a = 1, . . . , N) with classical action S[φ ]:3
S[φ ] =
 Z x 
 2
 1 ∂ μ
 φa (x) ∂μ φa (x) − m2
 2
 φa (x)φa (x) −
 4!N
 λ
 (φa (x)φa (x))2
 
 .
 (13)
Here
 summation
 over repeated indices is implied and we use the shorthand notation
R
x
 The ≡
 RC
 generating dx
0 R
 dd x with functional x ≡ (x0 W , x). [J, R] The for time connected path C Green’s will be specified functions later.
 in the presence of
two source terms ∼ Ja (x) and ∼ Rab (x, y) is given by
Z[J, R] = exp (iW [J, R])
=
 Z
 D φ exp i 
S[φ ] + Z
x
Ja (x)φa (x) +
 1
 2 Z
xy
 Rab(x, y)φa (x)φb (y)
 .(14)
We define the macroscopic field φa and the connected two-point function Gab by varia-
tion of W in the presence of the source terms:
δ δ W J a(x)
 [J, R]
 = φa(x) ,
 δ δ Rab W [J, (x, R]
 y) =
 1 2
 
φa (x) φb(y) + Gab (x, y)
 .
 (15)
Before constructing the 2PI effective action, we consider first the 1PI effective action. It
is obtained by a Legendre transform with respect to the source term which is linear in
the field,
ΓR
 [φ ] = W [J, R] −
 Z
x δ δ W Ja [J, (x)
 R]
 Ja (x) = W [J, R] −
 Z
x
 φa(x)J a(x) .
 (16)
We note that:
1. ΓR≡0[φ ] corresponds to the standard 1PI effective action.
2. For R 6= 0 it can be formally viewed as the 1PI effective action for a theory governed
by the modified classical action, SR [φ ], in the presence of a non-constant “mass
term” ∼ Rab (x, y) quadratic in the fields:
SR
 [φ ] = S[φ ] +
 2
 1
 Z
xy
 Rab (x, y)φa(x)φb (y) .
 (17)
This can be directly observed from (14) and the fact that the standard 1PI effective action
is obtained from the same defining functional integral in the presence of a linear source
3 We always work in units where h̄ = c = 1 such that the mass of a particle is equal to its rest energy (mc2)
and also to its inverse Compton wavelength (mc/h̄).
Introduction to Nonequilibrium Quantum Field Theory
 12
term only. As a consequence, it is straightforward to recover for ΓR [φ ] all “textbook”
relations for the 1PI effective action taking into account R. For instance, ΓR [φ ] to one-
loop order is given by
i
ΓR(1loop) [φ ] = S R [φ ] + 2
 Tr ln[G−1
 0 (φ ) − iR] ,
 (18)
which is the familiar result for the 1PI effective action with S[φ ] → SR [φ ] and G−1
 0 (φ ) →
G0 −1
(φ ) − iR. Similarly, one finds that the exact inverse propagator is obtained by second
functional field differentiation,
δ 2 ΓR [φ ]
= iG−1
 ab (x, y)
δ φa (x)δ φb (y)
= i h
G−1
 0,ab
 (x,
 y)
 −
 iR
ab
 (x,
 y)
 −
 Σ
R
 ab
 (x,
 y)
 i
 .
 (19)
Here the classical inverse propagator iG−1
 0,ab (x, y; φ ) = δ 2
 S[φ ]/δ φa (x)δ φb (y) reads
iG−1
 0,ab(x, y; φ )
 = − 
 x + m2
 +
 6N
 λ
 φc (x)φc (x)
 δab δ (x − y)
λ
−
 3N
 φa (x)φb (x)δ (x − y) ,
 (20)
and ΣRab (x, y) is the proper self-energy, to which only one-particle irreducible Feynman
diagrams contribute, i.e. diagrams which cannot be separated by opening one line. These
relations will be used below.
We now perform a further Legendre transform of ΓR [φ ] with respect to the source R
in order to arrive at the 2PI effective action:
Γ[φ , G] = ΓR
 [φ ] −
 Z
xy δ δ Rab ΓR (x, [ φ ]
 y)
 Rba(y, x)
|δ δ Rab W {z [J, (x, R]
 y) }
 =
 1 2
 
φa(x)φb (y) + Gab (x, y)

= ΓR [φ ] −
 1
 2 Z
 xy
 Rab(x, y)φa (x)φb(y) − 2
 1
 TrR G .
 (21)
Here we have used that the relation between φ and J is R-dependent, i.e. inverting
φ = δ W [J, R]/ δ J yields J = J R (φ ). From the above definition of ΓR[φ ] one therefore
finds
δ Rab δ ΓR[φ (x, ]
 y)
 =
 δ δ Rab W [J, (x, R]
 y)
 +
 Z
z δδ W J [J, c (z) R] δ Rab δ Jc (x, (z)
 y)
 −
 Z
z
 φc (z)
 δ Rab δ Jc(z)
 (x, y)
δ W [J, R]
=
 .
 (22)
δ Rab (x, y)
Introduction to Nonequilibrium Quantum Field Theory
 13
Of course, the two subsequent Legendre transforms which have been used to arrive at
(21) agree with a simultaneous Legendre transform of W [J, R] with respect to both source
terms:
Γ[φ , G] = W [J, R] −
 Z
x δ δ W Ja [J, (x)
 R]
 Ja (x) −
 Z
xy δ δ Rab W [J, (x, R]
 y)
 Rab(x, y)
= W [J, R] − Z
x
 φa (x)J a(x) −
 2 1
 Z
xy
 Rab (x, y)φa(x)φb (y) − 1
 2
 Tr G R (23)
From this one directly observes the stationarity conditions:
δ δ Γ[φ φa (x)
 , G]
 = −J a(x) − Z
y
 Rab (x, y)φb (y) ,
 (24)
δ Γ[ φ , G]
 1
δ Gab (x, y)
 = − 2
 Rab (x, y) ,
 (25)
which give the equations of motion for φ and G in the absence of the sources, i.e. J = 0
and R = 0.
To get familiar with Eq. (21) or (23), we may directly calculate Γ[φ , G] to one-loop
order using the above results for ΓR [φ ]. Plugging (18) into (21) one finds to this order
Γ[φ , G] ≃ S[φ ] + 2
 i
 Tr ln 
G−1
 0 (φ ) − iR − 1
 2
 TrR G .
 (26)
If we set G−1 = G−1
 0 (φ ) − iR then we can write
i
 i
Γ[φ , G] ≃ S[φ ] + 2
 Trln G−1 + 2
 Tr G−1
 0 G + const ,
 (27)
with Tr G−1G ∼ Tr 1 = const. To verify this one can check from the stationarity condition
(25) that indeed to this order
δ Γ[φ , G]
 i
 i
 1
δ G
 ≃ − 2
 G−1 + 2
 G−1
 0 (φ ) = − 2
 R
 ⇒
 G−1 = G−1
 0 (φ ) − iR .
To go beyond this order it is convenient to write the exact Γ[φ , G] as the one-loop type
expression (27) and a ‘rest’:
i
 i
Γ[ φ , G] = S[φ ] + Tr ln G−1 + Tr G−1
 0 (φ ) G + Γ2 [φ , G] + const
2
 2
(28)
Here we have added an irrelevant constant which can be adjusted for normalization. To
get an understanding of the ‘rest’ term Γ2[φ , G] we vary this expression with respect to
G, which yields
G−1
 ab (x, y) = G0,ab −1
 (x, y; φ ) − iRab (x, y) − Σab(x, y; φ , G) ,
 (29)
Introduction to Nonequilibrium Quantum Field Theory
 14
where we have written
δ Γ2 [φ , G]
Σab (x, y; φ , G) ≡ 2i
 δ Gab(x, y)
(30)
Comparing with the exact expression (19) one observes that
Σab (x, y; φ , G) = ΣRab (x, y; φ ) ,
 (31)
which relates the functional G-derivative of Γ2 [φ , G] to the proper self-energy. As men-
tioned above, to the proper self-energy only 1PI diagrams contribute with propagator
lines associated to the effective classical propagator (G−1
 0 − iR)−1
 . The mapping be-
tween Σab (x, y; φ , G) and ΣRab(x, y; φ ) is provided by (29), which can be used to express
the full propagator G as an infinite series in terms of the classical propagator G0 and Σ:
G = (G−1
 0 − iR)
−1
 + (G−1
 0 − iR)−1
 Σ (G0 −1
 − iR)
−1
+(G−1
 0 − iR)−1
 Σ (G0 −1
 − iR)−1
 Σ (G0 −1
 − iR)
−1
 + ... ,
 (32)
where we employ an obvious matrix notation. As a consequence there is a direct corre-
spondence between a 1PI diagram with propagator lines associated to G and an infinite
set of 1PI diagrams with propagator lines associated to the classical propagator G0 . Be-
low we will consider some explicit examples. Most importantly, from the fact that to
Σ(φ , G) only 1PI diagrams contribute one can conclude that
•
 Γ2[φ , G] only contains contributions from two–particle irreducible (2PI) diagrams.
A diagram is said to be two-particle irreducible if it does not become disconnected
by opening two lines. Suppose Γ 2 [φ , G] had a two–particle reducible contribution.
The latter could be written as Γ̃GGΓ̃′, where GG denotes in a matrix notation two
propagator lines connecting two parts Γ̃ and Γ̃′ of a diagram. Then Σ(φ , G) would contain
a contribution of the form Γ̃GΓ̃′ since it is given by a derivative of Γ2 with respect to G.
Such a structure is one-particle reducible and cannot occur for the proper self-energy.
Therefore two-particle reducible contributions to Γ2 [φ , G] are absent.
Diagrammatically, the graphs contributing to Σ(φ , G) are obtained by opening one
propagator line in graphs contributing to Γ2 [φ , G]. This is exemplified for a two- and a
three-loop graph for Γ2 and the corresponding self-energy graphs:
Γ2 :
 −→
 Σ ∼ δ Γ2/δ G:
Because of (32) each 2PI diagram corresponds to infinite series of 1PI diagrams, and the
above 2PI two- and three-loop diagrams contain e.g. the full series of so-called “daisies”
and “ladder” resummations:
Introduction to Nonequilibrium Quantum Field Theory
 15
G 0
G 0
+
 +
 +
2.1. Loop or coupling expansion of the 2PI effective action
Loop or coupling expansions of the 2PI effective action proceed along the same lines
as the corresponding expansions for the standard 1PI effective action, with the only
difference that
•
 the full propagator G is associated to propagator lines of a diagram
•
 and only 2PI graphs are kept.
To be specific, for the considered N-component scalar field theory the diagrams are
constructed in the standard way from the effective interaction
iSint [φ , φ ] = −
 Z
x
 i
 6N
 λ
 φa (x) φa (x)φb (x)φb(x) −
 Z
x
 i
 4!N
 λ 
φa(x)φa (x)2
 ,
 (33)
which is obtained from the classical action (13) by shifting φa (x) → φa (x) + φa (x)
and collecting all terms cubic and quartic in the fluctuating field φa(x). As for the
1PI effective action, in addition to the quartic interaction there is an effective cubic
interaction for non-vanishing field expectation value, i.e. φa(x) 6= 0.
Since Γ[φ , G] is a functional, which associates a number to the fields φ and G, only
closed loop diagrams can appear. We consider for a moment the real scalar theory for
N = 1. To lowest order one has Γ2 [φ , G] = 0 and we recover the one-loop result given in
Eq. (27). At two-loop order there are two contributions
Γ2
 (2loop)
 [φ , G]
 = −i 3 
 1
 − i
 4! λ 
 Z
x
 λ
 G2 (x, x)
 λ
−i 6
 2 Zxy
 
 − i 6
 φ (x) − i 6
 φ (y)
G3 (x, y) ,
 (34)
where we have made explicit the different factors coming from the overall −i in the
defining functional integral for Γ[φ , G] (cf. Eq. (14)), the combinatorics and the vertices.
The 2PI loop expansion exhibits of course much less topologically distinct diagrams
than the respective 1PI expansion. For instance, in the symmetric phase (φ = 0) one
finds that up to four loops only a single diagram contributes at each order. At fifth order
there are two distinct diagrams which are shown along with the lower-loop graphs in
Fig. 1. For later discussion, we give here also the results up to five loops for general N.
For φ = 0 by virtue of O(N) rotations the propagator can be taken to be diagonal:
Gab (x, y) = G(x, y)δab .
 (35)
Introduction to Nonequilibrium Quantum Field Theory
 16
+
 +
 +
 +
FIGURE 1. Topologically distinct diagrams in the 2PI loop expansion up to five-loop order for φ = 0.
The suppressed prefactors are given in Eq. (37).
One finds to five-loop order:
5
Γ2
 (5loop)
 [G]|Gab=Gδab
 =
 ∑ Γ2
 (l)
 ,
 (36)
l=2
Γ2
 (2)
 = −
 λ 8
 (N 3
 + 2)
 Z
x
 G2 (x, x) , Γ2 (3)
 =
 iλ 48 2 (N 3N
 + 2)
 Z
xy
 G4(x, y) ,
Γ2
 (4)
 =
 λ 48
 3 (N + 27N
 2)(N 2
 + 8)
 Z
xyz
 G2 (x, y)G2 (x, z)G2(z, y) ,
Γ2
 (5)
 = −
 128
 iλ 4
 (N + 2)(N 81N 2
 + 3
 6N + 20)
 Z
xyzw
 G2 (x, y)G2(y, z)G2 (z, w)G2(w, x)
−
 iλ 32
 4 (N + 2)(5N 81N 3
 + 22)
 Z
xyzw
 G2 (x, y)G(x, z)G(x, w)G2(z, w)G(y, z)G(y, w) .
2.2. Renormalization
The 2PI resummed effective action Γ[φ , G(φ )] is defined in a standard way by suitable
regularization, as e.g. lattice regularization or dimensional regularization, and renormal-
ization conditions which specify the field theory to be considered. We employ renormal-
ization conditions for the two-point function, Γ(2) , and four-point function, Γ(4), given
by
δ 2 Γ[φ , G(φ )]
Γ(2) (x, y) ≡
 δ φ (x)δ φ (y) φ =0
 ,
 (37)
(4)
 δ 4 Γ[φ , G(φ )]
Γ (x, y, z, w) ≡
 δ φ (x)δ φ (y)δ φ (z)δ φ (w)
 φ =0
 ,
 (38)
where we consider a one-component field for notational simplicity. Without loss of
generality we use renormalization conditions for φ = 0 which in Fourier space read:
Z Γ(2)(p2 )| p=0 = −m2 R ,
 (39)
d
Z dp
2 Γ(2)(p2 )| p=0 = −1 ,
 (40)
Z2 Γ(4) (p1, p2 , p3)| p1 =p2 =p3 =0 = −λR ,
 (41)
Introduction to Nonequilibrium Quantum Field Theory
 17
with the wave function renormalization Z.4 Here the renormalized mass parameter mR
corresponds to the inverse correlation length. The physical four-vertex at zero momen-
tum is given by λR.
2.2.1. 2PI renormalization scheme to order λR 2
In order to impose the renormalization conditions (39)–(41) one first has to calculate
the solution for the two-point field G(φ ) for φ = 0, which encodes the resummation and
which is obtained from the stationarity condition for the 2PI effective action (25).
The renormalized field is
φR = Z −1/2 φ .
 (42)
It is convenient to introduce the counterterms relating the bare and renormalized vari-
ables in a standard way with
Zm2 = m2 R + δ m2 ,
 Z 2 λ = λR + δ λ ,
 δ Z = Z − 1 ,
 (43)
and we write
G(φ ) = ZGR (φR) .
 (44)
In terms of the renormalized quantities the classical action (13) reads
S =
 Zx 
 1
 2
 ∂μ φR ∂ μ
 φR − 1 2
 mR 2 φR 2 − λR 24
 φR 4 + 1
 2
 δ Z ∂ μ φR ∂ μ
 φR − 1 2
 δ m2φR 2 −
 δλ 24 φR
 4
 
 . (45)
Similarly, one can write for the one-loop part Trln G−1 = Tr ln G−1
 R up to an irrelevant
constant and with G = G(φ ):
2
 i
 Tr G−1
 0
 (
φ
 )G(
φ
 )
 =
 −
 1
 λR 2 Z
 + x
 δ 
 λ1
 x
 +
 m
2
 R
 +
 δ
 Z
1
 
x
 +
 δ
 m
1
 2
 
 GR (x, y; φR )| x=y
−
 4
 Z
x
 φR 2(x)GR (x, x; φR ) .
 (46)
Here δ Z1, δ m21 and δ λ1 denote the same counterterms as introduced in (43), however,
approximated to the given order. To express Γ2 in terms of renormalized quantities it is
useful to note the identity
Γ2 [φ , G(φ )]|λ = Γ2[φR , G R (φR )]|λR +δ λ ,
 (47)
which simply follows from the standard relation between the number of vertices, lines
and fields by counting factors of Z. Therefore, one can replace in Γ2 the bare field and
4
 For the Fourier transform of the n-th derivative one has
Γ(n)(x1 , . . . xn) =
 Z
 (2π d4 p1 )4
 e
−ip1 x1
 ...
 Z
 (2π d4 pn )4
 e
−ip nxn
 (2π )4 δ 4 (p1 + . . . pn )Γ(n) (p1 , . . . pn ) .
Introduction to Nonequilibrium Quantum Field Theory
18
propagator by the renormalized ones if one replaces bare by renormalized vertices as
well. We emphasize that mass and wavefunction renormalization counterterms, δ Z and
δ m2 , do not appear explicitly in Γ2 . The counterterms in the classical action (45), in
the one-loop term (46) and beyond one-loop contained in Γ2 have to be calculated for a
given approximation of Γ2 . For an explicit example, we consider here the 2PI effective
action to order λR 2 with
Γ2[φR , GR (φR )]=− 8
 1
 λR Z
x
 G2 R(x, x; φR ) + i 12
 1
 λR 2 Z
xy
 φR (x)G3 R (x, y; φR )φR (y)
+i 48
 1 λR
 2
 Z
xy
 GR 4
 (x, y; φR) − 8
 1
 δ λ2 Z
x
 G2 R(x, x; φR ) ,
(48)
where the last term contains the respective coupling counterterm at two-loop. There are
no three-loop counterterms since the divergences arising from the three-loop contribu-
tion in (48) are taken into account by the lower counterterms.
One first has to calculate the solution GR ( φR ) obtained from the stationarity condition
(25) for the 2PI effective action. For this one has to impose the same renormalization
condition as for the propagator (39) in Fourier space:
iG−1
 R (p2
 ; φR )| p=0,φR=0 = −mR 2
 ,
for given finite renormalized “four-point” field
(49)
δ 2iG−1
 R (x, y; φR )
V R (x, y; z, w) ≡
 δ φR(z)φR (w)
 φR=0
 .
(50)
For the above approximation we note the identity
δ 2 Γ[φR, GR (φR )]
δ φR (x)φR(y)
 φR =0
 ≡ iG−1
 R (x, y; φR )|φR=0
(51)
for
δ Z = δ Z1
 ,
 δ m2 = δ m21
 ,
 δ λ1 = δ λ2 ,
 (52)
such that (49) for GR is trivially fulfilled because of (39). In contrast to the exact theory,
for the 2PI effective action to order λR 2 a similar identity does not connect the proper
four-vertex with VR .5 Here the respective condition for the four-point field VR in Fourier
space reads
V R (p1, p2 , p3)| p1 =p2 =p3 =0 = −λR .
 (53)
Note that this has to be the same than for the four-vertex (41). For the universality
class of the φ 4 theory there are only two independent input parameters, which we
5
 We emphasize that for more general approximations the equation (51) may only be valid up to higher
order corrections as well. This is a typical property of self-consistent resummations, and it does not affect
the renormalizability of the theory. In this case the proper renormalization procedure still involves, in
particular, the conditions (49) and (53).
Introduction to Nonequilibrium Quantum Field Theory
 19
take to be mR and λR , and for the exact theory VR and the four-vertex agree. The
renormalization conditions (39)–(41) for the propagator and four-vertex, together with
the scheme (49)–(53) provides an efficient fixing of all the above counter terms. In
particular, it can be very conveniently implemented numerically, which turns out to be
crucial for calculations beyond order λR .
We emphasize that the approximation (48) for the 2PI effective action can only be ex-
pected to be valid for sufficiently small φR ≪ mR/gR 3 .2 If the latter 4 is 4 not fulfilled there are
additional O(λR 2) contributions at three-loop ∼ λ R φR and ∼ λR φR . This approximation
should therefore not be used to study the theory in the spontaneously broken phase or
near the critical temperature of the second-order phase transition. Quantitative studies of
the latter can be performed in practice using 1/N expansion of the 2PI effective action
to NLO.
2.2.2. Renormalized equations for the two- and four-point functions
From the 2PI effective action to order λR 2 we find with δ λ1 = δ λ2 from (52) for the
two-point function:
iG−1
(x,
 R
 y;
 φ
R
 )
 =
 −
h
 1
 (1 + δ Z1 )x + m2 R + δ m21
+ 2
 i
 (λR 2 + 2 δ λ1 ) GR(x, x; φR ) + φ R 2(x) i
 2 i
 3 δ (x − y)
+ λR GR (x, y; φR )φR (x) φR (y) + λR GR (x, y; φR ) .
 (54)
2
 6
According to (51) this expression coincides with the one for the propagator
δ 2 Γ[φR, GR (φR )]/δ φR(x)δ φR (y) at φR = 0. It is straightforward to verify this using
δ Γ[φR , GR (φR )] δ Γ[φR , GR ]
δ φR (x)
 ≡
 δ φR (x)
 ,
 (55)
which is valid since the variation of GR (φR ) with φR does not contribute due to the
stationarity condition (25). The four-point field (50) in this approximation is given by
V R (x, y; z, w) = −(λR + δ λ1 )δ (x − y)δ (x − z)δ (x − w)
1
 δ 2 GR (x, x; φR )
− 2
 (λR + δ λ1)
 δ φR (z)δ φR(w) φR =0
 δ (x − y)
i
+ 2
 λR 2 δ (x − w)δ (y − z) + δ (y − w)δ (x − z)
+
 δ δφR 2 GR (z)δ (x, φR(w) y; φR)
 φR =0
 !
G2 R(x, y; φR = 0) .
 (56)
Introduction to Nonequilibrium Quantum Field Theory
 20
Inserting the chain rule formula
δ δ 2 φR iGR (z)δ (x, φR y; (w)
 φR )
 φR =0
 = −
 Z
u,v
 GR(x, u; φR )V R (u, v; z, w)GR (v, y; φR )
 φR =0
 (57)
one observes that Eqs. (56) and (54) form a closed set of equations for the determination
of the counterterms δ Z1 , δ m21 and δ λ1 . Together with (52) one notes that δ λ would be
undetermined from these equations alone. This counterterm is determined by taking into
account the equation for the physical four-vertex, which is obtained from the above 2PI
effective action as
δ 4 Γ[φR, GR ( φR )]
δ φR (x)δ φR (y)δ φR(z)δ φR(w)
 φR=0
 = −(λR + δ λ )δ (x − y)δ (x − z)δ (x − w)
+V R (x, y; z, w) +V R(x, z; y, w) +VR (x, w; y, z)
−2
 δ iG−1
(x,
 δ R
 GR(z, y;
 w)
 φ
R
 )
 +
 δ
 iG
 δ R
 GR −1
(x,
 (y, z;
 w)
 φ
R
 )
 +
 δ
 iG
 δ R
 −1
 GR (x,
 (y, w;
 z)
 φ
R
 )
 !
 φR =0
 ,
 (58)
where from (54) one uses the relation
δ iG−1
 R (x, y; φR)
 1
δ GR (z, w)
 = − 2
 ( λR + δ λ1)δ (x − y)δ (x − z)δ (x − w)
φR =0
i
+ 2
 λR 2G2 R (x, y; φR = 0)δ (x − z)δ (y − w) .
 (59)
We emphasize that the counterterm δ λ plays a crucial role in the broken phase, since
it is always multiplied by the field φ0 and hence it is essential for the determination of
the effective potential. It is also required in the symmetric phase, in particular, when one
calculates the momentum-dependent four-vertex using Eq. (58).
It is instructive to consider for a moment the 2PI effective action to two-loop order
for which much can be discussed analytically. In this case Z = 1 and the renormalized
vacuum mass mR of Eq. (39) or, equivalently, of Eq. (49) is given to this order by
m2 λ
 R
 =
 2
 1 μ
 mR
 ε
 Z
 2
 (2π dd 1
k
 )d
 k
2
 mR +
 m
R
 21
 
−1
 +
 m2
 m2
 λ
= −
 16π 2 
 ε
 − ln
 μ̄
 +
 2
 
 +
 λ
 ,
 (60)
where m2 = m2 R + δ m21 and λ = λR + δ λ1 and we have used (52). Here we have employed
dimensional regularization and evaluated the integral in d = 4 − ε for Euclidean mo-
 εmenta k. The bare coupling 2 in the action 2 (13) has been rescaled accordingly, λ → μ λ ,
and is dimensionless; μ̄ ≡ 4π e−γE μ and γE denotes Euler’s constant. Below we will
consider lattice regularizations for comparison and to go beyond two-loop order.
Introduction to Nonequilibrium Quantum Field Theory
 21
Similarly, the zero-temperature four-point function resulting from Eq. (53) for the 2PI
effective action to order λR is given by
λR = λ −
 λ λ 2
 λR λR μ
 ε
 Z
 1
 (2π dd k
 )d
 mR
 k
 2
 +
 m
2 R
 
 −2
= λ −
 16π 2 
 ε
 − ln
 μ̄
 
 (61)
with (52). We emphasize that the same zero-temperature equation is obtained starting
from the renormalization condition for the proper four-vertex (41) with δ λ = 3δ λ1 .
One observes that all counterterms are uniquely fixed by the renormalization procedure
discussed in the previous section.
Though dimensional regularization is elegant for analytical computations, it turns
out that high momentum cutoff regularizations are often more efficient for numerical
implementations. We will discuss below cutoff regularizations that are obtained by
formulating the theory on a discrete space-time lattice. In particular, it is often convenient
to carry out the numerical calculations using unrenormalized equations, or equations
where only the dominant (quadratically) divergent contributions in the presence of
scalars are subtracted. In order to obtain results that are insensitive to variations of the
cutoff, it is then typically sufficient to consider momentum cutoffs that are sufficiently
large compared to the characteristic energy-momentum scale of the process of interest.
It should be also stressed that a number of important renormalized quantities such as
renormalized masses or damping rates can be directly inferred from the oscillation
frequency or the damping of the time-evolution for the unrenormalized propagator G.
2.3. 2PI effective action for fermions
The construction of the 2PI effective action for fermionic fields proceeds along very
similar lines than for bosons. However, one has to take into account the anti-commuting
(Grassmann) behavior of the fermion fields. The main differences compared to the
bosonic case can be already observed from the one-loop part (Γ2 ≡ 0) of the 2PI effective
action. For vanishing field expectation values it involves the integrals:
Fermions: −i ln
 Z
 D ψ̄ D ψ e iS0 ( f )
 = −i ln det ∆−1
 0
 
 1
 2 = −i i
 Tr ln ∆−1
 0 ,
Bosons:
 −i ln
 Z
 D φ eiS0 = −i ln det G−1
 0
 − =
 2
 Trln G−1
 0 ,
 (62)
Grassmann where S( 0 f )
 = fields. R
 d4 xd4 For y ψ̄ Dirac (x)i∆−1
 fermions 0 (x, y)ψ with (y) denotes mass m( a ffermion ) the free action inverse that propagator is bilinear reads
 in the
i∆−1
 0 (x, y) = [i∂
 / x − m( f )] δ (x − y) ,
 (63)
where ∂ / ≡ γ μ ∂μ with Dirac matrices γμ (μ = 0, . . . , 3), and ψ̄ = ψ † γ 0 . For the bosons S0
is given by the quadratic part of (13). Comparing the two integrals one observes that the
Introduction to Nonequilibrium Quantum Field Theory
 22
factor 1/2 for the bosonic case is replaced by −1 for the fermion fields because of their
anti-commuting property. With this difference, following along the lines of Sec. 2 one
finds that the 2PI effective action for fermions can be written in complete analogy to (28).
Accordingly, for the case of vanishing fermion field expectation values, hΨi = hΨ̄i = 0,
one has:
Γ[∆] = −i Tr ln ∆−1 −i Tr ∆−1
 0 ∆ + Γ2 [∆] + const
 (64)
Here Γ2 [∆] contains all 2PI diagrams with lines associated to the time ordered propa-
gator ∆(x, y) = hT Ψ(x)Ψ̄(y)i. As for the 1PI effective action diagrams get an additional
minus sign from each closed fermion loop. The trace “Tr” includes integration over time
and spatial coordinates, as well as summation over field indices.
As for the bosonic case of Eq. (25), the equation of motion for ∆ in absence of external
sources is obtained by extremizing the effective action:
δ Γ[∆]
δ ∆(x, y)
= 0 .
(65)
Using (64) this stationarity condition can be written as
∆−1 (x, y) = ∆−1
 0 (x, y) − Σ( f )
 (x, y; ∆) ,
with the proper fermion self-energy:
f ) δ Γ2 [∆]
Σ( (x, y; ∆) ≡ −i
 δ ∆(y, x)
(66)
(67)
2.3.1. Chiral quark-meson model
As an example we consider a quantum field theory involving two fermion flavors
(“quarks”) coupled in a chirally invariant way to a scalar σ –field and a triplet of pseu-
doscalar “pions” π a (a = 1, 2, 3). The classical action reads
S =
 Z
 d4 xn
ψ̄ i∂ h
 / ψ + 2
 1 
∂μ σa ∂ μ a σ + ∂μ π a ∂ 2μ π a
 
 2+ N f
 ψ̄ [σ + iγ5 τ π ] ψ −V (σ + π )o
 ,
 (68)
where π 2 ≡ π a π a . Here τ a denote the standard Pauli matrices and h/N f is the Yukawa
coupling. The number of fermion flavors is N f ≡ 2 and N2 f is the number of scalar
components. The above action is invariant under chiral SUL(2) × SU R(2) ∼ O(4) trans-
formations. For a quartic scalar self-interaction,
2 2 1
 2 2 2 λ
 2 2V (σ + π ) = 2
 m (σ + π ) +
 4!N 2
 f
 σ + π 2
 ,
 (69)
Introduction to Nonequilibrium Quantum Field Theory
 23
this model corresponds to the well known linear σ –model, incorporating the chiral
symmetry of massless two-flavor QCD.
Here we do not consider the possibility of a spontaneously broken symmetry, thus
field expectation values vanish. For this model the 2PI effective action is then a func-
tional of fermion as well as scalar propagators. The scalar fields form an O(4) vec-
tor φa (x) ≡ (σ (x), ~ π (x) ) and we denote the full scalar propagator by Gab (x, y) with
a, b = 0, . . ., 3. In addition to its Dirac structure the fermion propagator carries flavor
indices i, j for N f = 2 flavors. The 2PI effective action for this Yukawa model is given
by
i
 i
Γ[G, ∆] = 2
 Trln G−1 + 2
 TrG−1
 0 G −i Tr ln ∆
−1
 − i Tr ∆−1
 0 ∆ + Γ2 [G, ∆] + const ,
 (70)
with the free scalar and fermion inverse propagators
iG−1
 0,ab (x, y) = −( x + m2
)δ (x − y)δab , i∆0,i −1
 j (x, y) = (i∂
 /x − m( f ) ) δ (x − y)δi j . (71)
The equation of motions for the scalar and fermion propagators are obtained from the
stationary conditions
δ Γ[G, ∆]
 δ Γ[G, ∆]
= 0
 ,
 = 0 .
 (72)
δ Gab (x, y)
 δ ∆i j (x, y)
The first non-zero order in a loop expansion of Γ2 [G, ∆] consists of a purely scalar
contribution corresponding to the two-loop diagram of Fig. (1), as well as a fermion-
scalar contribution depicted here graphically:
solid line: fermion propagator ∆
dashed line: scalar propagator G
Without loss of generality in the absence of spontaneous chiral symmetry breaking, the
effective action and its functional derivatives can be evaluated for Gab taken to be the
unit matrix in O(4)–space. Similarly, the most general fermion two-point function can
be taken to be proportional to unity in flavor space and we can write:
Gab (x, y) = G(x, y) δab
 ,
 ∆i j (x, y) = ∆(x, y) δi j .
 (73)
The two-loop approximation then reads
Γ2
 (2loop)
 [G, ∆]|Gab=Gδab ,∆i j =∆δi j
 = −
 λ 8
 (N 2
 f 3
 + 2)
 Z
x
 G2(x, x)
−ih
2 N 2 f
 Z
xy
 tr[∆(x, y) ∆(y, x)]G(x, y) ,
 (74)
where the trace “tr” acts only in Dirac space. According to (30) and (67) the self-energies
can then be obtained by
Σab (x, y) = 2i
 Z
x′ y′ δ δ Γ2 G(x |Gδab ′ , y ,∆δi ′
) j δ δ G G(x′ ab
 (x, , y′ y)
 )
 =
 2i
 δab N f 2 δ Γ2 δ G(x, |Gδab,∆δi y)
 j
 ,
 (75)
Introduction to Nonequilibrium Quantum Field Theory
 24
Σi ( j f )
 (x, y) = −i
 δab N f δ Γ2 δ ∆(y, |Gδab x)
 ,∆δi j
 ,
 (76)
where we have used G = Gab δ ab /N f 2 with δaa = N 2 f , and equivalently for the fermion
propagator with δii = N f .
2.4. Two-particle irreducible 1/N expansion
In this section we discuss a systematic nonperturbative approximation scheme for the
2PI effective action. It classifies the contributions to the 2PI effective action according
to their scaling with powers of 1/N, where N denotes the number of field components:
Γ2[φ , G] = ΓLO
 2 [φ , G] + Γ2
 NLO
 [φ , G] + ΓNNLO
[φ 2
 , G] + . . .
∼ N 1
 ∼ N 0
 ∼ N −1
Each subsequent contribution ΓLO
 2 , Γ2
 NLO , ΓNNLO 2
 etc. is down by an additional factor
of 1/N. The importance of an expansion in powers of 1/N stems from the fact that
it provides a controlled expansion parameter that is not based on weak couplings. It
can be applied to describe physics characterized by nonperturbatively large fluctuations,
such as encountered near second-order phase transitions in thermal equilibrium, or for
extreme nonequilibrium phenomena such as parametric resonance. For the latter cases a
2PI coupling or loop expansion is not applicable. The method can be applied to bosonic
or fermionic theories alike if a suitable field number parameter is available, and we
exemplify it here for the case of the scalar O(N)-symmetric theory with classical action
(13). We comment on its application to the chiral quark-meson model below.
2.4.1. Classification of diagrams
We present a simple classification scheme based on O(N)–invariants which
parametrize the 2PI diagrams contributing to Γ[φ , G]. The interaction term of the
classical action in Eq. (13) is written such that S[φ ] scales proportional to N. From the
fields φa alone one can construct only one independent invariant under O(N) rotations,
which can be taken as tr φ φ ≡ φ 2 = φaφa 2 ∼ N. The minimum φ0 of the classical effective
potential for this theory is given by φ0 = N(−6m2 /λ ) for negative mass-squared m2
and scales proportional to N. Similarly, the trace with respect to the field indices of the
classical propagator G0 is of order N.
The 2PI effective action is a singlet under O(N) rotations and parametrized by the two
fields φa and Gab . To write down the possible O(N) invariants, which can be constructed
from these fields, we note that the number of φ –fields has to be even in order to construct
an O(N)–singlet. For a compact notation we use (φ φ ) ab = φa φb . All functions of φ and
G, which are singlets under O(N), can be built from the irreducible (i.e. nonfactorizable
in field-index space) invariants
φ 2,
 tr(G n )
 and
 tr(φ φ Gn ).
 (77)
Introduction to Nonequilibrium Quantum Field Theory
 25
+
a
b
+
a
b
+
a
a
+
b
b
FIGURE 2. Graphical representation of the φ –dependent contributions for Γ2 ≡ 0. The crosses denote
field insertions ∼ φa φa for the left figure, which contributes at leading order, and ∼ φa φb for the right
figure contributing at next-to-leading order.
We note that for given N only the invariants with n ≤ N are irreducible — there cannot
be more independent invariants than fields. We will see below that for lower orders in
the 1/N expansion and for sufficiently large N one has n < N. In particular, for the next-
to-leading order approximation one finds that only invariants with n ≤ 2 appear, which
makes the expansion scheme appealing from a practical point of view.
Since each single graph contributing to Γ[φ , G] is an O(N)–singlet, we can express
them with the help of the set of invariants in Eq. (77). The factors of N in a given graph
have two origins:
•
 each irreducible invariant is taken to scale proportional to N since it contains exactly
one trace over the field indices,
• while each vertex provides a factor of 1/N.
The expression (28) for the 2PI effective action contains, besides the classical action,
the one-loop contribution proportional to Tr ln G−1 + Tr G−1
 0 (φ )G and a nonvanishing
Γ2[φ , G] if higher loops are taken into account. The one-loop term contains both leading
order (LO) and next-to-leading order (NLO) contributions. The logarithmic term cor-
responds, in absence of other terms, simply to the free field effective action and scales
proportional to the number of field components N. To separate the LO and NLO con-
tributions at the one-loop level consider the second term Tr G−1
 0 (φ )G. From the form of
the classical propagator (20) one observes that it can be decomposed into a term pro-
portional to tr(G) ∼ N and terms ∼ (λ /6N) [tr(φ φ ) tr(G) + 2 tr( φ φ G)]. This can be seen
as the sum of two “2PI one-loop graphs” with field insertion ∼ φa φa and ∼ φa φb , re-
spectively, as shown in Fig. 2. Counting the factors of N coming from the traces and the
prefactor, one sees that only the first contributes at LO, while the second one is NLO.
According to the above rules one draws all topologically distinct 2PI diagrams and
counts the number of closed lines as well as the number of lines connecting two field in-
sertions in a diagram following the indices. For instance, the left diagram of Fig. 2 admits
one line connecting two field insertions and one closed line. In contrast, the right figure
admits only one line connecting two field insertions by following the indices. Therefore,
the right graph exhibits one factor of N less and becomes subleading. Similarly, for the
two-loop graph below one finds:
a
 a
 a
 b
 a
 b
+
 +
b
 b
 a
 b
 b
 a
LO
 ;
 NLO
Introduction to Nonequilibrium Quantum Field Theory
 26
(trG)2/N ∼ N
 trG2 /N ∼ N 0
The same can be applied to higher orders. We consider first the contributions to
Γ2[φ = 0, G] ≡ Γ2 [G], i.e. for a vanishing field expectation value and discuss φ 6= 0 be-
low. The LO contribution to Γ2 [G] consists of only one two-loop graph, whereas to NLO
there is an infinite series of contributions which can be analytically summed:
ΓLO
 2 [G]
 = −
 4!N
 λ
 Z
x
 Gaa (x, x)Gbb (x, x) ,
 (78)
i
ΓNLO
 2
 [G] =
 Tr ln[ B(G) ] ,
 (79)
2
λ
B(x, y; G) = δ (x − y) + i
 6N
 Gab (x, y)Gab (x, y) .
 (80)
In order to see that (79) with (80) sums the following infinite series of diagrams
+
 +
 +
 +
one can expand:
Trln[ B(G) ]=
−
+Zx
 
i 6N
 λ
 Gab (x, x)Gab (x, x)

1
 2 Zxy 
i 6N
 λ
 Gab (x, y)Gab (x, y)

i
 6N λ
 Ga′ b
′ (y, x)Ga′ b′ (y, x)

...
(81)
The first term on the r.h.s. of (81) corresponds to the two-loop graph with the index
structure exhibiting one trace such that the contribution scales as trG2/N ∼ N 0 . One
 0
observes that each additional contribution scales as well proportional to (trG2 /N)n ∼ Nfor all n ≥ 2. Thus all terms contribute at the same order.
The terms appearing in the presence of a nonvanishing field expectation value are
obtained from the effectively cubic interaction term in (33) for φ 6= 0. One first notes
that there is no φ -dependent graph contributing at LO. To NLO there is again an infinite
series of diagrams ∼ N 0 which can be summed:
ΓLO
 2 [φ , G] ≡ Γ2 LO
[G] ,
 (82)
ΓNLO
[φ 2
 , G]
 =
 ΓNLO
[φ
 2
 ≡ 0, G] +
 6N
 iλ
 Z
xy
 I(x, y; G) φa(x) Gab (x, y) φ b (y) ,
 (83)
I(x, y; G) =
 6N
 λ
 Gab (x, y)Gab (x, y) − i
 6N
 λ
 Z
z
 I(x, z; G)Gab(z, y)Gab(z, y) . (84)
Introduction to Nonequilibrium Quantum Field Theory
 27
The series of terms contained in (83) with (84) corresponds to the diagrams:
+
 +
 +
 +
+
 +
 +
+
 +
The functions I(x, y; G) and the inverse of B(x, y; G) are closely related by
B−1(x, y; G) = δ (x − y) − iI(x, y; G) ,
 (85)
which follows from convoluting (80) with B−1 and using (84). We note that B and I do
not depend on φ , and Γ2 [φ , G] is only quadratic in φ at NLO.
The 2PI can be straightforwardly applied to other theories as well. For instance,
the chiral quark-meson model of Sec. 2.3.1 admits a 1/N f expansion. For the latter
the LO contribution scales ∼ N f 2 and is given by a purely scalar two-loop term as in
Eq. (74). Note that this two-loop contribution contains a LO part ∼ N 2 f as well as a
NNLO part ∼ N 0 f . At NLO the first fermion contributions to Γ2 appear, which scale
∼ N f (cf. Eq. (74)). If no small coupling is available the expansion parameter N f ≡ 2 in
this case might not be suitable for a quantitative estimate at low orders. See however the
precision tests of Sec. 5 for the scalar O(N) model, which exhibit a good convergence
already for moderate values of N & 2.
2.4.2. Symmetries and validity of Goldstone’s theorem
We emphasize that by construction each order in the 2PI 1/N–expansion respects
O(N) symmetry. In particular, this is crucial for the validity of Ward identities. For the
case of spontaneous symmetry breaking note that Goldstone’s theorem is fulfilled at any
order in the 2PI 1/N expansion. According to (77) the 2PI effective action can be written
as a function of the O(N) invariants:
In the case of spontaneous Γ[φ symmetry , G] ≡ Γ 
 breaking φ 2, tr(Gn), one tr(φ has φ G a constant p )
 .
 φ 0 and the propa-
 (86)
6=gator can be parametrized as
Gab (φ ) = GL ( φ 2)P ab
 L
 + GT (φ 2 )P ab
 T
 ,
 (87)
where P ab
 L = φa φb
 /φ 2 and PT ab
 = δab
 − PL ab
 are the longitudinal and transverse projectors
with respect to the field direction. The 2PI resummed effective action Γ[φ , G(φ )] is
obtained by evaluation at the stationary value (25) for G. The mass matrix M ab can
then be obtained from
δ 2 Γ[φ , G(φ )]
Mab ∼
 δ φa δ φb
 φ =const
 .
 (88)
If Γ[φ , G(φ )] is calculated from (86) and (87) one observes that indeed it depends only
on one invariant, φ 2 : Γ[φ , G(φ )] = Γ[φ 2 ]. The form of the mass matrix Mab can now be
Introduction to Nonequilibrium Quantum Field Theory
 28
inferred straightforwardly. To obtain the effective potential U (φ 2/2), we write
Γ[φ 2 ]
 φ =const
 = Ωd+1U (φ 2/2),
 (89)
where Ωd+1 is the d + 1 dimensional Euclidean volume. The expectation value of the
field is given by the solution of the stationarity equation (24) which becomes
∂ U (φ 2 /2)
= φa U ′ = 0,
 (90)
∂ φa
where U ′ ≡ ∂ U /∂ (φ 2 /2) and similarly for higher derivatives. The mass matrix reads
2
 ∂ 2U (φ 2 /2)
 L
 T
Mab
 =
 = δabU ′ + φa φbU ′′ = (U ′ + φ 2U ′′)P ab
 +U ′ P ab
.
 (91)
∂ φa ∂ φb
In the symmetric phase (φa = 0) one finds that all modes have equal mass squared
Mab
 2 = U ′δab
 . In the broken phase, with φa
 6= 0, Eq. (90) implies that the mass of the
transverse modes ∼ U ′ vanishes identically in agreement with Goldstone’s theorem.
Truncations of the 2PI effective action may not show manifestly the presence of massless
transverse modes if one considers the solution of the stationarity equation (24) for the
two-point field G encoding the resummation. It is important to realize that the 2PI
resummed effective action Γ[φ , G(φ )] complies fully with the symmetries.
3. NONEQUILIBRIUM QUANTUM FIELD THEORY
Out of equilibrium dynamics requires the specification of an initial state. This may
include a density matrix at a given time ρD (t0 = 0) in a mixed (TrρD 2 (0) < 1) or pure state
(TrρD 2 (0) = 1). Nonequilibrium means that the initial density matrix does not correspond
to a thermal equilibrium density matrix: ρD(0) 6= ρD (eq)
 with for instance ρD (eq)
 ∼ e−β H
for the case of a canonical thermal ensemble. Once the initial state is specified, the
dynamics can be described in terms of a functional path integral with the classical
action S as employed in the previous sections above. The corresponding nonequilibrium
effective action is the generating functional for all correlation functions with the initial
correlations determined by ρD(0).
3.1. Nonequilibrium generating functional
All information about nonequilibrium quantum field theory is contained in the
nonequilibrium generating functional for correlation functions:
Z[J, R; ρD] = Tr n
ρD (0) T C ei(R
x J(x)Φ(x)+ 1 2 R
xy R(x,y)Φ(x)Φ(y)) o
 .
 (92)
Here T C denotes
 time-ordering
 along the time path C appearing in the source term inte-
grals with R
x ≡ RC dx0 R
 dd x as specified below. In complete analogy to Eq. (14) we have
Introduction to Nonequilibrium Quantum Field Theory
 29
introduced the generating functional with two source terms, J and R, in order to construct
the corresponding nonequilibrium 2PI effective action below. We suppress field indices
in the notation, which can be directly recovered from (14). The nonequilibrium cor-
relation functions, i.e. expectation values of time-ordered products of Heisenberg field
operators Φ(x), are obtained by functional differentiation. For instance the two-point
function reads (TrρD = 1):
δ 2 Z[J, R; ρD]
Tr{ρ D(0) T C Φ(x)Φ(y)} ≡ hT C Φ(x)Φ(y)i =
 iδ J(x) iδ J(y)
 J=R=0
 .
 (93)
There is a simple functional integral representation of Z[J, R; ρD ] similar to (14). One
writes the trace as
Z[J, R; ρD] =
 Z
 dφ (1) (x)dφ (2)(x) hφ (1)|ρD(0)|φ (2)i
hφ (2) |Tei(
R
x J(x)Φ(x)+ 1 2 R
xyR(x,y)Φ(x)Φ(y)
) |φ (1)i ,
 (94)
where the matrix elements are taken with respect to eigenstates of the Heisenberg field
operators at intial time, Φ(t = 0, x)|φ (i)i = φ (i)(x)|φ (i) i, i = 1, 2. The source-dependent
matrix element can be expressed in terms of a path integral using standard techniques if
one considers a finite, closed real-time contour C :
t
Contour time ordering along this path corresponds to usual time ordering along the
forward piece C + and antitemporal ordering on the backward piece C −. Note that any
time on C − is considered later than any time on C + . One can then use
hφ (2) |Tei(
R
x J(x)Φ(x)+ 1 2 R
xy R(x,y)Φ(x)Φ(y)
) |φ (1) i
φ (2) (x)=φ (0− ,x)
=
 (1) Z
 D ′ φ ei(S[φ ]+
R
x J(x)φ (x)+ 2 1 R
xy R(x,y)φ (x)φ (y)
 ) ,
 (95)
φ (x)=φ (0+ ,x)
which is the same relation as employed to obtain standard path integral expressions for
vacuum or equilibrium matrix elements. Note that the closed time contour starting from
the initial time t0 = 0 and ending at t0 is required because the density matrix element
hφ (1)|ρD(0)|φ (2) i is taken on both sides with respect to states at time t0 . Using (95)
in (94) one finds that the expression for Z[J, R; ρD ] directly displays the ingredients
entering nonequilibrium quantum field theory — the quantum fluctuations described
by the functional integral with action S, and the statistical fluctuations encoded in the
Introduction to Nonequilibrium Quantum Field Theory
 30
weighted average with the initial-time elements:
φ
 (2)
Z[J, R; ρD] =
 Z
 dφ (1) dφ (2) hφ (1)|ρD(0)|φ (2) i
 Z
 (1)
 D ′ φ ei(S[φ ]+
R
x J(x)φ (x)+ 1 2 R
xyR(x,y)φ (x)φ (y)
)
φ|
 initial conditions
 {z
 } |
 quantum {z
 dynamics
 (96)
 }
Of course, this generating functional can be equally applied to standard vacuum physics.
In this case the closed time contour just ensures the normalization Z|J=R=0 = 1, which
can simplify calculations. In particular, the density matrix is time independent for the
vacuum as well as for the thermal equilibrium case. In contrast to thermal equilibrium,
the nonequilibrium density matrix cannot be interpreted as an evolution operator in
imaginary time, as is possible for instance for the canonical ρ (eq) ∼ e− β H .
In the literature formulations of closed time path generating functionals often exhibit
an infinite time interval ] −∞, ∞[. In this case the number of field labels has to be doubled
to distinguish the fields on the underlying closed contour. Causality implies that for
any n-point function with finite time arguments contributions of an infinite time path
cancel for times exceeding the largest time argument of the n-point function. To avoid
unnecessary cancellations of infinite time path contributions we always consider finite
time paths. The largest time of the path is kept as a parameter and is evolved in the time
evolution equations as described below. We stress that the initial time of the path has to
be finite — a system that can thermalize will be already in equilibrium at any finite time
if the initial time is sent to the infinite past.
3.2. Initial conditions
To understand in more detail how the initial density matrix enters calculations, we
consider first the example of a Gaussian density matrix whose most general form can be
written as
hφ (1) |ρ D(0)|φ (2)i =
2πξ 1
 2
 exp n
iφ̇ (φ (1) − φ (2) )−
 σ 8ξ 2 + 2
 1 h(
φ
 (1)
 −
 φ
 )
2
 +(
φ
 (2)
 −
 φ
 )
2
 i
+i
 p
 η (φ (1)
 φ )2 (φ (2) φ ) 2 +
 σ 2 − 1
 (
φ
 (1)
 φ
 )(
φ
 (2)
 φ
 )
 ,
 (97)
2ξ
 h − − − i 4ξ 2
 −
 −
 o
where we neglect the spatial dependencies for a moment. Note that for homogeneous
field expectation values taking into account spatial dependencies simply amounts to
adding a momentum label in Fourier space (cf. also Sec. 3.4.3). In order to see that
this is the most general Gaussian density matrix, we first note that (97) is equivalent to
the following set of initial conditions for one- and two-point functions:
φ = Tr {ρD(0)Φ(t)} |t=0
 ,
 φ̇ = Tr{ρD(0)∂ t Φ(t)} |t=0 ,
 (98)
Introduction to Nonequilibrium Quantum Field Theory
 31
ξ ξ η 2 = = Tr 2
 1 Tr 
ρD ρD (0)Φ(t)Φ(t (0) ∂t Φ(t)Φ(t ′) |t=t ′) ′=0 + Φ(t)∂t − φ φ ,
′ Φ(t ′ )
 |t=t ′ =0
 φ̇ φ ,
 (99)
2 σ
 2
 
 
 −η + 4ξ
 2 = Tr 
ρD (0)∂t Φ(t)∂t ′ Φ(t ′) |t=t ′ =0 − φ̇ φ̇ .
It is straightforward to check explicitly the equivalence between the initial density matrix
and the initial conditions for the correlators:
TrρD (0) =
 Z−∞
 ∞
1
 dφ h φ |ρD(0)|φ ∞
 i
 1
= 2πξ 2 Z−∞
 ∞
 dφ exp n
 − 2ξ
 2 (φ − φ )2 o
 = 1 ,
 (100)
Tr (0)Φ(0)} = p
 1
 dφ φ exp 1
 ( φ φ )2
 φ →φ = +φ
 φ , (101)
{ρD 2πξ 2 Z−∞
 n
 − 2 ξ
 2 − oetc. Similarly, one finds since p
 only Gaussian integrations appear that all initial n-point
functions with n > 2 are given in terms of the one- and two-point functions. Note also
that the anti-symmetrized initial correlator involving the commutator of Φ and ∂t Φ is
not independent because of the field commutation relation.
The crucial observation is that higher initial time derivatives are not independent as
can be observed from the exact field equation of motion, which reads for the real scalar
Φ4-theory:
2 λ
h∂t Φi = −m2 hΦi −
 6N
 hΦ3 i .
 (102)
Since hΦ3i is given in terms of one- and two-point functions for Gaussian ρD(0) also
second and higher time derivatives are not independent. We conclude that the most
general Gaussian density matrix is indeed described by the five parameters appearing
in (97). In particular, all observable information contained in the density matrix can be
conveniently expressed in terms of correlation functions and their derivatives (98) and
(99).
For further interpretation of the initial conditions we note that
Tr ρD 2 (0) =
 Z−∞
 ∞
 d φ
 Z−∞
 ∞
 dφ ′ hφ |ρD(0)|φ ′ ihφ ′ |ρD (0)|φ i =
 σ
 1
 .
 (103)
The latter shows that for σ > 1 the density matrix describes a mixed state. A pure state
requires σ = 1 or, equivalently, using (99) this condition can be expressed in terms of
initial-time correlators as
hTr 
ρ D(0)Φ(t)Φ(t1 ′
) |t=t′ =0 − φ φ ihTr 
ρD (0)∂t Φ(t)∂ t ′ Φ(t ′
 ) |t=t ′=0 − φ̇ φ̇
 i
 (104)
 1
− 
 2
 Tr 
ρD (0) ∂ t Φ(t)Φ(t ′) + Φ(t)∂ t ′ Φ(t ′)
 |t=t ′=0 − φ̇ φ
 2
 =
 4
 .
Introduction to Nonequilibrium Quantum Field Theory
 32
In field theory such a Gaussian initial condition is associated with a vanishing initial
particle number, as will be discussed in Sec. 4. For σ = 1 the “mixing term” in (97) is
absent and one obtains a pure-state density matrix of the product form:
ρD(0) = |ΩihΩ|
 (105)
with Schrödinger wave function
hφ |Ωi =
 (2πξ 1
 2)1/4
 exp
 n
i
φ̇
 φ
 −
  4ξ 1
 2
 +
 i
 2ξ
 η 
 (
φ
 −
 φ
 )
2
o
 .
 (106)
In order to go beyond Gaussian initial density matrices one can generalize the above
example and parametrize the most general density matrix as
hφ (1) | ρD (0)|φ (2) i = N ei f C [φ ] ,
 (107)
with normalization factor N and f C [φ ] expanded in powers of the fields:
fC [φ ] = α0 + Z
x
 α1(x)φ (x) +
 2 1
 Z
xy
 α2 (x, y)φ (x)φ (y) +
 3!
 1
 Z
xyz
 α3(x, y, z)φ (x)φ (y)φ (z)
+
 4! 1
 Z
xyzw
 α4(x, y, z, w)φ (x)φ (y)φ (z)φ (w) + . . .
 (108)
Here φ (0 + , x) = φ (1) (x) and φ (0− , x) = φ (2) (x). We emphasize that (108) employs a
compact notation: since the density matrix ρD (0) is specified at initial time t0 = 0, all
time integrals contribute only at time t0 of the closed time contour. As a consequence, the
coefficients α1(x), α2 (x, y), α3 (x, y, z), . . . vanish identically for times different than t0 .
For instance, up to quadratic order one has
Z
x
 α1 (x)φ (x) ≡
 Zx n
α1 (1)
 (1,1)
 (x)φ (1) (x) + α1 (2)
 (1) (x) φ (2) (x) (1,2)
 o
 ,
 (2)Z
 xy
 α2(x, y)φ (x)φ (y) ≡
 Zxy
 n
α2 (x, y)φ (1)(x)φ (y) + α2 (x, y)φ (1)(x)φ (y)
+α2 (2,1)
 (x, y)φ (2) (x)φ (1)(y) + α2 (2,2)
(x, y)φ (2) (x)φ (2)(y)o
 .
Note that α0 is an irrelevant normalization constant. For a physical density matrix the
other (2,2)∗
 coefficients (1,2)
 are of course (2,1)∗
 not arbitrary. Hermiticity implies α1 (1)
 = −α1 (2)∗
 , α2 (1,1)
 =
−α2
 and α2 = −α2
 , which can be directly compared to the discussion above.
3.3. Nonequilibrium 2PI effective action
Using the parametrization (107) and (108) for the most general initial density matrix,
one observes that the generating functional (96) introduced above can be written as
Z[J, R; ρD] =
 Z
 D φ e i(S[φ ]+
R
x J(x)φ (x)+ 1 2 R
xy R(x,y)φ (x)φ (y)+ 3! 1 R
xyz α3(x,y,z)φ (x)φ (y)φ (z)+...
) .
(109)
Introduction to Nonequilibrium Quantum Field Theory
 33
Here we have neglected an irrelevant normalization constant and rescaled the sources in
(96) as J(x) → J(x) − α1 (x) and R(x, y) → R(x, y) − α2 (x, y). The sources can therefore
be conveniently used to absorb the lower linear and quadratic contributions coming from
the density matrix specifying the initial state. This absorbtion of α1 in J and α2 in R
exploits the fact that the initial density matrix is completely encoded in terms of initial-
time sources α1, α2 , α3 , . . . for the functional integral.
The generating functional (109) can be used to describe situations involving arbitrarily
complex initial density matrices. However, often the initial conditions of an experiment
may be described by only a few lowest n–point functions. For many practical purposes
the initial density matrix is well described by a Gaussian one. For instance, the initial
conditions for the reheating dynamics in the early universe at the end of inflation are de-
scribed by a Gaussian density matrix to high accuracy. Clearly, the subsequent evolution
builds up higher correlations which are crucial for the process of thermalization. These
have to be taken into account by the quantum dynamics, which is not approximated by
the specification of an initial density matrix!
From (109) one observes that for Gaussian initial density matrices, for which α3 =
α4 = . . . = 0, one has
Z h
J, R; ρD
 (gauss)
i
 ≡ Z[J, R] .
 (110)
As a consequence, in this case the nonequilibrium generating functional corresponds
to the 2PI generating functional introduced in Eq. (14) for a closed time path. We can
therefore directly take over all steps from Sec. 2 in order to construct the nonequilibrium
2PI effective action and obtain the important result:
Nonequilibrium 2PI effective action = Γ[φ , G] with closed time path C
All previous discussions of Sec. 2 remain unchanged except that for nonequilibrium
the time integrals involve a closed time path! We emphasize again that the use of the
nonequilibrium 2PI effective action represents no approximation for the dynamics —
higher irreducible correlations can build up corresponding to a non-Gaussian density
matrix for times t > t0 . It only restricts the “experimental” setup described by the
initial conditions for correlation functions. Non-Gaussian initial density matrices pose
no problems in principle but require taking into account additional initial-time sources.
This is most efficiently described in terms of nPI effective actions for n > 2.
3.4. Exact evolution equations
Without approximation the 2PI effective action Γ[φ , G] contains the complete infor-
mation about the quantum theory. The functional representation of the nonequilibrium
Γ[φ , G] employs a one-point (φ ) and a two-point field (G), whose physical values have to
be computed for all times of interest. The equations of motion for these fields are given
by the stationarity conditions (24) and (25).
We will consider first the scalar field theory in the symmetric regime, where
Γ[φ = 0, G] ≡ Γ[G] is sufficient. We come back to the case of a nonvanishing field
expectation value as well as to fermionic and gauge fields below. The equation of
Introduction to Nonequilibrium Quantum Field Theory
 34
motion (29) for G reads
G−1(x, y) = G−1
 0 (x, y) − Σ(x, y; G) − iR(x, y) ,
 (111)
where the self-energy Σ(x, y; G) is given by (30) The form of the equation (111) is
suitable for boundary value problems as e.g. appear for thermal equilibrium. However,
nonequilibrium time evolution is an initial value problem. The equation of motion can
be rewritten as a partial differential equation suitable for initial-value problems by
convolution with G, using Rz G−1 (x, z)G(z, y) = δ (x − y):
Z
z
 G−1
 0 (x, z)G(z, y) −
 Z
z
 [Σ(x, z) + iR(x, y)]G(z, y) = δ (x − y) .
 (112)
Here we employ the notation δ (x − y) ≡ δC (x0 − y0 ) δ (x 2
 − y). For the scalar theory the
evolution classical propagator equation for (cf. the Sec. time-ordered 2) is G−1
 0 (x propagator:
 − y) = i  x + m  δ (x − y) and one obtains the
 x + m2 
 G(x, y) + i Z
z
 [Σ(x, z; G) + iR(x, y)]G(z, y) = −iδ (x − y)
 (113)
3.4.1. Spectral and statistical components
In the following we introduce a decomposition of the two-point function G into spec-
tral and statistical components. The corresponding evolution equations for the spectral
function and statistical propagator are fully equivalent to the evolution equation for G,
but have a simple physical interpretation. While the spectral function encodes the spec-
trum of the theory, the statistical propagator gives information about occupation num-
bers. Loosely speaking, the decomposition makes explicit what states are available and
how often they are occupied.
For the real scalar field theory there are two independent real–valued two–point
functions, which can be associated to the expectation value of the commutator and the
anti-commutator of two fields,
commutator:
 ρ (x, y) = ih[Φ(x), Φ(y)]i ,
 (114)
1
anti-commutator:
 F(x, y) = 2
 h{Φ(x), Φ(y)}i .
 (115)
Here ρ (x, y) denotes the spectral function and F(x, y) the statistical two-point function.
The decomposition identity for spectral and statistical components of the propagator
reads:
i
G(x, y) = F(x, y) − 2
 ρ (x, y) signC (x 0 − y0 )
 (116)
The identity is easily understood by making the time-ordering for the propagator ex-
plicit:
G(x, y) = hΦ(x)Φ(y)iΘC (x0 − y0) + hΦ(y)Φ(x)iΘC (y0 − x0 )
Introduction to Nonequilibrium Quantum Field Theory
35
=
 2
 1
 i
 h{Φ(x), Φ(y)}i ΘC (x0 − y0 ) + ΘC (y0 − x0)

−
 2
 ih[Φ(x), Φ(y)]i ΘC (x0 sign − y0 C ) (x0 − ΘCy0 (y0 ) .
 − x0)

|
 {z
 − }
For real scalar fields the real functions obey F(x, y) = F(y, x) and ρ (x, y) = −ρ (y, x).
We note from (114) that the spectral function ρ encodes the equal-time commutation
relations
ρ (x, y)|x0 =y0 = 0 , ∂x0 ρ (x, y)|x0 =y0 = δ (x − y) .
 (117)
The spectral function is also directly related to the retarded propagator ρ (x, y)Θ(x0 −y0 ),
or the advanced one − ρ (x, y)Θ(y0 − x0 ). However, it is important to realize that out
of equilibrium there are only two independent two-point functions — no more. These
can be associated to F and ρ , which has the advantage compared to e.g. an advanced
propagator that the non-analyticity entering through time-ordering is always explicit.
To obtain a similar decomposition for the self-energy, we separate Σ in a “local” and
“nonlocal” part according to
Σ(x, y; G) = −iΣ(0) (x; G)δ (x − y) + Σ(x, y; G) .
 (118)
Since Σ(0) just corresponds to a space-time dependent mass-shift it is convenient for the
following to introduce the notation
M 2 (x; G) = m2 + Σ(0) (x; G) .
 (119)
To make the time-ordering for the non-local part of the self-energy, Σ(x, y; G), explicit
we can use the same identity as for the propagator (116) to decompose:
i
Σ(x, y) = ΣF (x, y) − 2
 Σρ (x, y) signC (x0 − y0 ) .
 (120)
Though the discussion is given here for a vanishing field expectation value, it should
be emphasized that the same decompositions (116) and (120) apply also for φ 6= 0. The
r.h.s. of (115) would then receive an additional contribution subtracting the disconnected
part ∼ φ φ . Equivalently, one can always view (116) as defining F and ρ from the
connected propagator G irrespective of the value of φ .
We emphasize that the equivalent decomposition can be done for fermionic degrees of
freedom as well. The fermion propagator in terms of spectral and statistical components
reads in the same way as for bosons:
i
∆(x, y) = F ( f ) (x, y) − 2
 ρ ( f )(x, y) signC (x0 − y0 ) .
 (121)
However, in contrast to bosons for fermions the field anti-commutator corresponds to
the spectral function,
anti-commutator:
 ρ ( f ) (x, y) = ih{Ψ(x), Ψ̄(y)}i ,
 (122)
( f ) 1
commutator:
 F (x, y) = 2
 h[Ψ(x), Ψ̄(y)]i .
 (123)
Introduction to Nonequilibrium Quantum Field Theory
 36
This can be directly observed from the time-ordered fermion propagator ∆(x, y) =
hΨ(x)Ψ̄(y)iΘC (x0 − y0 ) − hΨ̄(y)Ψ(x)iΘC (y0 − x0 ). Here the minus sign is a conse-
quence of the anti-commutation property for fermionic fields (cf. Sec. 2.3). In analogy
to the bosonic case, the equal-time anti-commutation relations for the fields are again
encoded in ρ ( f )(x, y). For instance, for Dirac fermions one has
γ 0 ρ (x, y)|x0 =y0 = iδ (x − y)
 (124)
with the Dirac matrix γ 0 , and the two-point functions have the hermiticity properties
( ρ (y, x) ) † = − γ 0 ρ (x, y)γ 0 ,
 ( F(y, x) )† = γ 0F(x, y)γ 0 .
 (125)
The corresponding decomposition for the fermion self-energy reads:6
f ( f )
 i ( f )
 0Σ( )(x, y) = ΣF (x, y) − 2
 Σρ (x, y) signC (x0 − y ) .
 (126)
3.4.2. Detour: Thermal equilibrium
To see the above decomposition in a probably more familiar context, we consider
for a moment thermal equilibrium. This is done for illustrational purposes only, and
we emphasize that the notion of an equilibrium temperature is nowhere implemented in
nonequilibrium quantum field theory. If a nonequilibrium evolution approaches thermal
equilibrium at late times then this a prediction of the theory and not put in by hand.
The 2PI effective action in thermal equilibrium is given by the same expression (28)
if the closed time path is replaced by an imaginary path C = [0, −iβ ]. Here β denotes
the inverse temperature. Since thermal equilibrium is translation invariant, the two-point
functions depend only on relative coordinates and it is convenient to consider the Fourier
transforms F (eq) (ω , p) and ρ (eq) (ω , p) with
F
 (eq)
 (x, y) =
 Z
 (2π dω dd ) d+1
 p e
−iω (x0 −y0 )+ip(x−y) F (eq)
 (ω , p)
 (127)
and equivalently for the thermal spectral function.
The periodicity (“KMS”) condition characterizing thermal equilibrium for the prop-
agator in imaginary time is given by G(x, y)|x0 =0 = G(x, y)|x0=−iβ . Employing the de-
composition identity (116) for the propagator G, one can write the periodicity condition
as
F (eq)
 (ω , p) + 2
 i ρ (eq)
 (ω , p) = e
−β ω
 
 F (eq)
 (ω , p) − 2
 i ρ (eq)
 (ω , p)
 .
 (128)
6
 If there is a local contribution to the proper self-energy, one separates in complete analogy to the scalar
equation (118). The decomposition (126) is taken for the non-local part of the self-energy, while the local
contribution gives rise to an effective space-time dependent fermion mass term.
Introduction to Nonequilibrium Quantum Field Theory
 37
To see this note that x0 = 0 comes first on the imaginary path C = [0, −iβ ], while
x0 = −iβ comes latest such that signC (x0 − y0 ) contributes opposite signs on the left
and on the right of the equation. Eq. (128) can be rewritten in a more standard form as a
fluctuation-dissipation relation for bosons:7
F
 (eq)
 (ω , p) = −i
 
 2
 1
 + nBE (ω )
 ρ (eq) (ω , p)
 (129)
with nBE(ω ) = (eβ ω − 1)−1 denoting the Bose-Einstein distribution (eq) function. Eq. (129)
relates the spectral function to the statistical propagator. While ρ encodes the in-
formation about the spectrum of the theory, one observes from (129) that the function
(eq)F encodes the statistical aspects in terms of the particle distribution function nBE .
In the same way one obtains for the Fourier transforms of the spectral and statistical
components of the self-energy the thermal equilibrium relation
ΣF (eq)
 (ω , p) = −i
 
 2
 1
 + nBE(ω )
 Σρ (eq)
 (ω , p) .
 (130)
We note that the ratio Σρ (eq)
 (ω , p)/2ω plays in the limit of a vanishing ω -dependence
the role of the decay rate for one-particle excited states with momentum p.
For fermions the anti-periodicity condition of the fermionic propagator in ther-
mal equilibrium, ∆(x, y)|x0=0 = −∆(x, y)|x0 =−iβ , implies a corresponding fluctuation-
dissipation relation. The difference is that the Bose-Einstein distribution in (129) is
replaced by the Fermi-Dirac distribution nFD(ω ) = (eβ ω + 1)−1 according to 1/2 +
nBE (ω ) → 1/2 − nFD (ω ) in the respective relation.
It is important to realize that out of equilibrium F and ρ are not related by the
fluctuation-dissipation relation! In contrast to the nonequilibrium theory, the relation
(129) is a manifestation of the tremendous simplification that happens if the system is
in thermal equilibrium. An even more stringent reduction occurs for the vacuum where
nBE (ω ) ≡ 0. In this respect, nonequilibrium quantum field theory is more complicated
since it admits the description of more general situations. Of course, the nonequilibrium
theory encompasses the thermal equilibrium or vacuum theory as special cases. We leave
now this equilibrium detour and return to the nonequilibrium case.
3.4.3. Nonequilibrium evolution equations
Out of equilibrium we have to follow the time-evolution both for the statistical propa-
gator, F, as well as for the spectral function, ρ . The evolution equations are obtained
from (113) with the help of the identities (116) and (120). Most importantly, once
expressed in terms of F and ρ the time-ordering is explicit and the respective sign-
7
 In our conventions the Fourier transform of the real-valued antisymmetric function ρ (x, y) is purely
imaginary.
Introduction to Nonequilibrium Quantum Field Theory
 38
functions appearing in the time-ordered propagator can be conveniently evaluated along
the time contour C .
With the notation (118) the time evolution equation for the time-ordered propagator
(113) reads

x + M2 (x; G)
 G(x, y) + i Z
z
 Σ(x, z; G)G(z, y) = −iδ (x − y) ,
 (131)
where we have set R ≡ 0. The influence of the initial-time sources encoded in R is
discussed
 below.
 For the evaluation along the time contour C involved in the integration
with R
z ≡ RC dz0
 R dd z we employ (116) and (120):
i
 Z
z
 i
 Σ(x, z; G)G(z, y) = i
 Zz
 n
ΣF (x, z)F(z, i
 y)
− 2
 ΣF (x, z) ρ (z, y) sign C (z0 − y0 )− 2
 Σρ (x, z)F(z, y) signC (x0 − z0)
 (132)
− 1 4
 Σρ
 (x, z)ρ (z, y) signC (x0 − z0)signC (z0 − y0 )o
 .
The first term on the r.h.s. vanishes because of integration along the closed time contour
C (cf. Sec. 3.1). To proceed for the second term one splits the contour integral such that
the sign-functions have a definite value, for instance
Z
C
 dz0
 signC (z0
 − y0
) =
 Z 0
y0
 dz0
 (−1) +
 Zy0
 0
 dz0
 = −2
 Z0
y0
 dz0
 (133)
for the closed contour with initial time t0 = 0. To evaluate the last term on the r.h.s. of
Eq. (132) it is convenient to distinguish the cases
(a) ΘC (x0 − y0) = 1:
Z
C
 dz0
 signC (x0
 − z0
 )signC (z 0
 − y0
 ) =
 Z0
y0
 dz0
(−1) +
 Zy0
 x0
 dz0
 +
 Zx0
 0
dz0 (−1) ,
 (134)
(b) ΘC (y0 − x0 ) = 1:
Z
C
 dz0
 signC (x0
 − z0
 )signC (z0
 − y0
 ) =
 Z0
 x0
 dz0
(−1) +
 Zx0
 y0
 dz0
 +
 Zy0
 0
dz0 (−1) .
 (135)
One observes that (a) and (b) differ only by an overall sign factor ∼ signC (x0 − y0 ).
Combining the integrals therefore gives:
i
 Z
z
 Σ(x, z; G)G(z, y) =
 Z
 dd z
 (Z0
 x
0
 dz0 Σρ (x, z)F(z, y) −
 Z0
y
0
 dz0 ΣF (x, z)ρ (z, y)
− 2
 i
 signC (x0 − y0 )
 Zy0
 x0
 dz0
 Σρ (x, z)ρ (z, y)
)
 .
 (136)
Introduction to Nonequilibrium Quantum Field Theory
 39
One finally employs
i
x G(x, y) =  x F(x, y) − 2
 signC (x0 − y 0 )x ρ (x, y) − iδ (x − y)
 (137)
such that the δ -term cancels with the respective one on the r.h.s. of the evolution equation
(131). Here we have used
− 2
 i
 ∂x 2 0 
ρ (x, y) signC (x0 − y0)
 = − 2
 i
 signC (x0 (x0 y0)∂ − y x0 0 )∂x ρ (x, 2 0 ρ y)
 (x, y)
−iδC −|
 −iδ (x {z
 − y) ,
 }
 0where (117) is employed for the last line and to observe that a term ∼ ρ (x, y)δC (x − y0 )
vanishes identically. Comparing coefficients, which here corresponds to separating real
and imaginary parts, one finds from
 (136)
 and (137)
 the equations for F(x, y) and ρ (x, y).
equations Using the abbreviated for the spectral notation function R t1 t2 dz and ≡ R the t1 t2 dz statistical 0 R−∞ ∞ dd z propagator:
 we arrive at the coupled evolution

 x + M 2 (x)
 ρ (x, y) = −
 Z
y0
 x0
 x0
 dz Σ ρ (x, z)ρ (z, y) ,
 y0

 x + M 2
 (x)
 F(x, y) = −
 Z0
 dz Σ ρ (x, z)F(z, y) +
 Z0
 dz ΣF (x, z)ρ (z, y).
(138)
These are causal equations with characteristic “memory” integrals, which integrate over
the time history of the evolution. We emphasize that the presence of memory integrals
is a property of the exact theory and in accordance with all symmetries, in particular
time reflection symmetry. The equations themselves do not single out a direction of
time and they should be clearly distinguished from phenomenological nonequilibrium
equations, where irreversibility is typically put in by hand. Since these equations are
exact they are fully equivalent to any kind of identity for the two-point functions such
as Schwinger-Dyson/Kadanoff-Baym equations without further approximations. For
φ = 0 the functional dependence of the self-energy corrections in (138) is given by
M 2 = M 2 (F), ΣF = ΣF (ρ , F) and Σρ = Σρ (ρ , F). The case φ 6= 0 is discussed below.
Note that the initial-time properties of the spectral function have to comply with
the equal-time commutation relations (117). In contrast, for F(x, y) as well as its first
derivatives the full initial conditions at t0 = 0 need to be supplied in order to solve
these equations. To make contact with the discussion of initial conditions in Sec. 3.2
and Eq. (99), we consider for
 a moment the spatially homogeneous case for which
for F(x, ρ y) (x, = y). F(x0 In terms , y0 ; x − of y) the = Fourier R [dd p/(2π components )d ] exp[ip(x F(t,t − ′; y)]F(x0 p) the , solution y0 ; p) and of equivalently
 the integro-
differential equations (138) requires the following initial conditions:
F(t,t ′; p)|t=t ′=0 ≡ ξp 2
 ,
 2
 1
 ∂t F(t,t ′; p) + ∂ t ′ F(t,t ′; p)
|t=t ′ =0 ≡ ξp ηp ,
Introduction to Nonequilibrium Quantum Field Theory
40
2 σp 2
∂ t ∂ t ′ F(t,t ′
; p)|t=t ′=0 ≡
 ηp +
 4ξp 2
 .
(139)
Here we have used that the required correlators at initial time are identical to those given
in Eq. (99) for the considered case φ ≡ 0, where we had suppressed the momentum
labels in the notation. Accordingly, these are the very same parameters that have to be
specified for the corresponding most general Gaussian initial density matrix (97). We
emphasize that the initial conditions for the spectral function equation are completely
fixed by the properties of the theory itself: the equal-time commutation relations (117)
specify ρ (t,t ′; p)| t=t ′=0 = 0, ∂t ρ (t,t ′; p)|t=t ′ =0 = 1 and ∂ t ∂t ′ ρ (t,t ′ ; p)|t=t ′=0 = 0 for the
anti-symmetric spectral function.
We are now in the position to discuss the role of the initial-time sources, which contain
the information about the initial-time density matrix. According to the discussion of
Secs. 3.2 and 3.3 the initial-time sources are fully described by the Fourier components
of the bilinear source term R(t,t ′; p) at t = t ′ = t0 = 0 for the case φ = 0. Mathematically,
the role of the initial-time sources for the evolution equations is rather simple: Since
these sources have support only for the initial time t0 and vanish identically for times
t 6= t0, they only fix the initial values for the correlators and their first derivatives.8 For
simpler differential equations this property is well documented in the literature on the
theory of Green’s functions. In order to see that this indeed holds also for the case
considered here, recall that the time evolution equations (138) are derived from (113)
for R ≡ 0. Eq. (139) shows that there is a one-to-one correspondence between the initial-
time sources parametrizing the density matrix (97) and the required initial conditions for
the solution of the time evolution equations. To check that no further dependencies on
R remain in the evolution equations
 for times t 6= t0, one notes that the R-dependence
a appears consequence, in (113) the as equation a term ∼governing R
z R(x, z)G(z, G(x, y), y) which (or F identically and ρ ) cannot vanishes explicitly for x0 depend 6= t0. As
 on
R for its time arguments different than t0, i.e. for all times of interest.
The clear separation of the dynamical role of spectral and statistical components
is a generic property of nonequilibrium field theory. As discussed in Sec. 3.4.2, the
nonequilibrium theory encompasses standard vacuum theory as a special case where this
separation is absent. This dichotomy for nonequilibrium time evolution equations is not
specific to scalar field degrees of freedom. In terms of spectral and statistical components
the equations for fermionic fields or gauge fields have a very similar structure as well. For
instance, the respective form of the evolution equations for the fermion spectral function
ρ ( f ) (x, y) and statistical propagator F ( f )(x, y) defined in (121) can be directly obtained
from (138) by the l.h.s. replacement:

x ( f ) + M 2
 ρ (x, y) −→ −[i∂ /x − m( f ) ] ρ ( f )(x, y)
 (140)
and equivalently for F (x, y). On the r.h.s. of (138) then appear the respective fermion
propagators and self-energies Σρ ( f )
 and ΣF ( f )
 as defined in (126). This can be directly
verified from the equation of motion for the time-ordered fermion propagator (66) by
8
 We consider additional external sources to be absent.
Introduction to Nonequilibrium Quantum Field Theory
41
convoluting it with ∆. For a free inverse propagator as in (71) for Dirac fermions this
yields
h
i∂ /x − m
( f )
 i
 ∆(x, y) − i Z
z
 Σ( f ) (x, z)∆(z, y) = iδC (x − y) .
 (141)
Following along the lines of the above discussion for scalars one finds for the fermion
case the coupled evolution equations:
h
i∂ /x − m
( f )
 i
 ρ ( f )
 (x, y) =
 Z
y0
 x0
 dz Σρ ( f )
(x, z)ρ ( f )(z, y) ,
h
i∂ /x − m
( f )
 i
 F ( f )
 (x, y) =
 Z
0
x0
 dz Σρ ( f )
(x, z)F ( f ) (z, y) −
 Z0
 y0
 dz ΣF ( f )
 (x, z)ρ ( f ) (z, y). (142)
Similarly, the nonequilibrium evolution equations for gauge fields can be obtained as
well. For instance, denoting the full gauge field propagator by
D μν (x, y) = F D μν
 (x, y) − 2
 i ρD μν
 (x, y) signC (x 0 − y0 )
 (143)
for a theory with free inverse gauge propagator given by
for covariant gauges iD−1
 with 0, μν (x, gauge-fixing y) = 
gμν  parameter − 1 − ξ
 −1
 ξ 
 and ∂μ ∂ν vanishing 
x δ (x − “background” y)
 fields,
 (144)
one finds the respective equations from (138) by

x + M2 
γν
 ρ (x, y) −→ − 
gμ γ  − (1 − ξ −1 )∂ μ ∂γ x ρD γν
 (x, y)
 (145)
and equivalently for F (x, y). Of course, the respective indices have to be attached
Dto the corresponding self-energies on the r.h.s. of the equations. The derivation of the
nonequilibrium gauge field evolution equations will be discussed in more detail in Sec. 6
in the context of higher nPI effective actions with n > 2.
3.4.4. Non-zero field expectation value
In the presence of a non-zero field expectation value, φ 6= 0, the form of the scalar
evolution equations for the spectral and statistical function (138) remain the same.
However, the functional dependence now includes M 2 = M2 (φ , F), ΣF = ΣF (φ , ρ , F)
and Σρ = Σρ (φ , ρ , F). Note that the local self-energy correction (119) described by
M 2 does not depend on the spectral function because the latter vanishes for equal-time
arguments. For the N-component scalar field theory (13) one has with φ 2 ≡ φa φa:
Mab(x; 2
 φ , F) =
 
 λ
 m2
 +
 6N
 λ 
F cc (x, x) + φ 2
 (x) 

 δ ab
+
 [F ab (x, x) + φa (x)φb (x)] ,
 (146)
3N
Introduction to Nonequilibrium Quantum Field Theory
 42
with the respective field indices attached to (138). In this case the evolution equations
for the spectral function and statistical two-point function (138) are supplemented by a
differential equation for φ given by the stationarity condition (24), which yields the field
evolution equation:

x +
 6N
 λ φ 2
 (x)
 δab + Mab 2
 (x; φ ≡ 0, F)
 φb (x) =
 δ δ φa Γ2
 (x)
 (147)
For comparison the corresponding equation for the classical field theory, δ S[φ ]/ δ φa (x),
is obtained from (147) by the replacement Mab (x; φ ≡ 0, F) → m2 δab and Γ2 ≡ 0. For
the above field evolution equation we have used that the one-loop type contribution to
the 2PI effective action reads
2
 i
 TrG0 −1
 (φ )G = −
 2 1
 Zx
 
x + m2
 +
 6N
 λ φ 2
 (x)
 δab +
 3N
 λ
 φa (x)φb (x)
 F ba (x, x) (148)
for the classical inverse propagator (20), such that one can write
2
 δ [iTrG−1
 0 (φ )G/2]
 2
m −
 δ φa (x)
 = Mab
 (x; φ ≡ 0, F) .
 (149)
The solution of (147) requires specifying the field and its first derivative at initial time.
In the context of the above discussion for homogeneous fields this just corresponds to
specifying (98), which together with the required initial conditions (139) for the two-
point function completes the correspondence with the initial-time density matrix (97).
3.4.5. Lorentz decomposition for fermion dynamics
The evolution equations for the fermion spectral function and statistical propagator
(142) are rather complicated in general. For Dirac fermions each two-point function con-
tains 16 complex components, which is often supplemented by additional field attributes
such as “flavor” or “color”. However, it is often not necessary in practice to consider all
components due to the presence of symmetries, which require certain components to
vanish identically without loss of generality. Depending on the symmetry properties of
the initial state and interaction these terms remain zero under the nonequilibrium time
evolution, which can dramatically simplify the analysis. It is typically very useful to
decompose the fields ρ ( f ) (x, y) and F ( f ) (x, y) of Eq. (142) into terms that have definite
transformation properties under Lorentz transformation. In order to ease the notation, in
this section we will write ρ ( f ) 7→ ρ , F ( f ) 7→ F, Σρ ( f )
 7→ A and ΣF ( f )
 7→ C. Using a standard
basis we write
μ
 μ
 1
 μν
ρ = ρS + iγ5ρP + γμ ρ V + γμ γ5 ρA + σμν ρ T ,
 (150)
2
where σ μν = 2 i [γμ , γν ] and γ5 = iγ 0γ 1 γ 2γ 3 . The 16 (pseudo-)scalar, (pseudo-)vector and
tensor components
ρS = tr  ̃ ρ ,
 ρP = −i tr  ̃ γ5ρ ,
 ρ V μ
 = tr  ̃ γ μ ρ ,
 ρA μ
 = tr  ̃ γ5γ μ ρ ,
 ρT μν
 = tr  ̃ σ μν ρ , (151)
Introduction to Nonequilibrium Quantum Field Theory
 43
are complex two-point functions. Here we have defined tr  ̃ ≡ 14 tr where the trace acts in
Dirac space. Equivalently, there are 16 complex components for F, A and C for given
other field attributes. Using the hermiticity properties (125) they obey
ρi (Γ)
 j (x, y) = − 
ρ ji (Γ)
 (y, x)
 ∗
 , F i(Γ)
 j (x, y) = 
Fji (Γ)
 (y, x)∗
 ,
 (152)
where Γ = {S, P,V, A, T}. Inserting the above decomposition into the evolution equa-
tions (142) one obtains the respective equations for the various components displayed
in (151).
For a more detailed discussion, we first consider the l.h.s. of the evolution equations
(142). In fact, the approximation of a vanishing r.h.s. corresponds to the standard mean–
field or Hartree–type approaches frequently discussed in the literature. However, to
discuss thermalization we have to go beyond such a “Gaussian” approximation: it is
crucial to include direct scattering which is described by the nonvanishing contributions
from the r.h.s. of the evolution equations. This is discussed in detail in Sec. 4. Starting
with the l.h.s. of (142) for the evolution equation for the spectral function one finds:
tr  ̃ tr  ̃ 
(i∂ / / − m m f f ) ) ρ 
 = = i∂μ ρ V μ 
 − μ m f m ρS f ,
 ,
−i tr  ̃ 
γ5 μ (i∂ − m f ρ 
 −i μ i∂μ ρA 
 i − νμ ρP m f V μ
 (153)
tr  ̃ 
γ μ (i∂ / / − m f ) ) ρ 
 = = (i∂ i μ ρS ) P ) + + 1
 i∂ν μνγδ ρT 
 − ρ ,
 m f μ
 ,
tr  ̃ 
γ5 γ μν
 (i∂ / − m f ) ρ 
 = (i∂ ρ μ
 V ν
 2
 ε ν V μ + i∂ν ρT,γδ μνγδ 
 − ρA δ m f μν
 .
The corresponding σ (i∂ − expressions ρ 
 −i for i∂ the ρ l.h.s. − i∂ of ρ the 
 evolution ε i∂γ equation ρA, 
 − for the ρT statistical
propagator F follow from (153) with the replacement ρ → F. We turn now to the inte-
grand on the r.h.s. of (142) for the evolution equation of ρ . For the various components
(151) one finds:
tr  ̃ [A ρ ] = AS ρS AP ρP + A V μ
 ρV,μ AA μ
 ρA,μ
− −+ 1 Aμν
 T ρT,μν ,
 (154)
2
tr  ̃ [γ5 A ρ ] = AS ρP + AP ρS iAV μ
 ρA,μ + iAA μ
 ρ V,μ
−i −+ 1
 ε μνγδ AT,μν ρT,γδ ,
 (155)
4
tr[
  ̃ γ μ
 A ρ ] = AS ρV μ + A V μ ρS − iAP ρ A μ + iA μ A ρP + iA V,ν ρT ν μ
tr  ̃ [γ5 γ μ
 A ρ ] = +iAT AS ρA μν
 μ
 + ρV,ν AA μ
 + ρS1
 2
 − εiAP μνγδ ρV μ
 AA,ν + iAV ρT,γδ μ
 ρP + + iAA,ν AT,νγ ρT
 ρA,δ νμ
 
 ,
 (156)
+iAT μν
 ρA,ν + 2
 1
 ε μνγδ AV,ν ρT,γδ + AT,νγ ρ V,δ 
 ,
 (157)
Introduction to Nonequilibrium Quantum Field Theory
 44
tr  ̃ [ σ μν A ρ ] = −i AS ρT A μν
 V μ + ρV ν
 AT − μν
 AV ρS ν ρV − μ
 2 1
 + ε μνγδ ε
 μνγδ
 AP AV,γ ρT,γδ ρA,δ + − AT,γδ AA,γ ρP
 ρV,δ
 
+i AA μ
 ρA ν − AA ν ρA μ  + i AT μγ
 ρT,γ ν − AT νγ
 ρT,γ μ .
 
 (158)
With the above expressions one obtains the evolution 
 equations for the 
 various Lorentz
components in a straightforward way using (142). We note that the convolutions ap-
pearing on the r.h.s. of the evolution equation (142) for F are of the same form than
those computed above for ρ . The respective r.h.s. can be read off Eqs. (154)–(158) by
replacing ρ → F for the first term and A → C for the second term under the integrals
of Eq. (142) for F. We have now all the relevant building blocks to discuss the most
general case of nonequilibrium fermionic fields. However, this is often not necessary in
practice due to the presence of symmetries, which require certain components to vanish
identically.
In the following, we will exploit symmetries of the action (68) for the chiral quark-
meson model described in Sec. 2.3.1 and of initial conditions in order to simplify the
fermionic evolution equations:
Spatial translation invariance and isotropy: We will consider spatially homogeneous
and isotropic initial conditions. In this case it is convenient to work in Fourier space and
we write
ρ (x, y) ≡ ρ (x0 , y0
 ; x − y) =
 Z
 (2π d3 p )3
 e
ip·(x−y)
ρ (x0 , y0 ; p) ,
 (159)
and similarly for the other two-point functions. Moreover, isotropy implies a reduction
of the number of independent two-point functions: e.g. the vector components of the
spectral function can be written as
ρV 0 (x0, y0 ; p) = ρV 0 (x0, y0 ; p) ,
 ~ ρ V (x0 , y0 ; p) = v ρV (x0 , y0; p) ,
 (160)
where p ≡ |p| and v = p/p.
Parity: The vector components ρV 0 (x0, y0 ; p) and ρ V (x0 , y0 ; p) are unchanged under a
parity transformation, whereas the corresponding axial-vector components get a minus
sign. Therefore, parity together with rotational invariance imply that
ρA 0(x0 , y0 ; p) = ρA (x0 , y0; p) = 0 .
 (161)
The same is true for the axial-vector components of F, A and C. Parity also implies the
pseudo-scalar components of the various two-point functions to vanish.
CP–invariance: For instance, under combined charge conjugation and parity transfor-
mation the vector component of ρ transforms as
ρV 0 (x0, y0 ; p) −→ ρ V 0 (y0 , x0 ; p) ,
ρV (x0, y0 ; p) −→ −ρ V (y0 , x0 ; p) ,
and similarly for AV 0 and A V . The F–components transform as
F V 0 (x0, y0 ; p) −→ −F V 0 (y0 , x0; p) ,
Introduction to Nonequilibrium Quantum Field Theory
 45
F V (x0, y0 ; p) −→ F V (y0 , x0; p) ,
0and similarly for C V and C V . Combining this with the hermiticity relations (152), one
obtains for these components that
Re ρ V 0 (x0 , y0; p)
 =
 Iρ V (x0 , y0; p) = 0 ,
Re F V 0 (x0 , y0; p)
 =
 IF V (x0 , y0; p) = 0 ,
(162)
Re A V 0 (x0 , y0; p)
 =
 IA V (x0 , y0; p) = 0 ,
ReC V 0 (x0 , y0 ; p)
 =
 IC V (x0 , y0 ; p) = 0 ,
for all times x0 and y0 and all individual momentum modes.
Chiral symmetry: The only components of the decomposition (150) allowed by chiral
symmetry are those which anticommute with γ5 . In particular, chiral symmetry forbids
a mass term for fermions (m f ≡ 0) and we have
ρS (x0, y0 ; p) = ρP (x0, y0 ; p) = ρT μν
 (x0 , y0 ; p) = 0 ,
 (163)
and similarly for the corresponding components of F, A and C. We emphasize, however,
that in the presence of spontaneous chiral symmetry breaking there is no such simplifi-
cation.
In conclusion, for the above symmetry properties we are left with only four indepen-
dent propagators: the two spectral functions ρ V 0 (x0 , y0; p) and ρ V (x0 , y0; p) and the two
corresponding statistical functions F V 0 (x0 , y0; p) and F V (x0, y0 ; p). They are either purely
real or imaginary and have definite symmetry properties under the exchange of their time
arguments x0 ↔ y0 . These properties as well as the corresponding ones for the various
components of the self-energy are summarized below:
ρ V 0 , AV 0 :
 imaginary, symmetric;
ρ V , AV :
 real, antisymmetric;
F 0, C 0 :
 imaginary, antisymmetric;
V VF V , C V :
 real, symmetric.
The exact evolution equations for the spectral functions read (cf. Eq. (142)):9
∂ 0 0 0
i
 ρ V
 (x , y ; p) = p ρ V (x0 , y0; p)
∂ x0+
 Zy0
 x0
 dz0 h
A V 0 (x0 , z0; p) ρ V 0 (z 0 , y0 ; p) − AV (x0 , z0; p) ρ V (z0 , y0 ; p)i
 ,
 (164)
∂
 0i
 0
 ρ V (x0 , y0 ; p) = p ρ V (x0 , y0; p)
∂ x
+
 Zy0
 x0
 dz0 h
A V 0 (x0 , z0
; p) ρ V (z 0 , y0
 ; p) − AV (x0 , z0
; p) ρ V 0 (z0 , y0
 ; p)i
 .
 (165)
9
We note that the following equations do not rely on the restrictions (162) imposed by CP–invariance:
they have the very same form for the case that all two-point functions are complex.
Introduction to Nonequilibrium Quantum Field Theory
 46
Similarly, for the statistical two-point functions we obtain (cf. Eq. (142)):
∂ 0 0 0
i
 0
 F V (x , y ; p) = p F V (x0 , y0 ; p)
∂ x
+
 Z0
 y0
 x0
 dz0 h
AV 0 (x0, z0 ; p) F V 0 (z0 , y0 ; p) − A V (x0 , z0 ; p) F V (z0, y0 ; p)
i
−
 Z0
 dz0
 h
 C V 0 (x0 , z0
; p) ρ V 0 (z 0 , y0
 ; p) −C V (x0 , z0
; p) ρV (z0, y0
 ; p)i
 ,
 (166)
∂
i
 0
 F V (x0 , y0 ; p) = p F V 0(x0 , y0 ; p)
∂ x
+
 Z0
 y0
 x0
 dz0 h
AV 0 (x0, z0 ; p) F V (z0, y0 ; p) − A V (x0 , z0; p) F V 0 (z0, y0 ; p)
i
−
 Z0
 dz0
 h
 C V 0 (x0 , z0
; p) ρ V (z 0 , y0
 ; p) −C V (x0 , z0
; p) ρV 0 (z0, y0
 ; p)i
 .
 (167)
The above equations are employed in Sec. 4 to calculate the nonequilibrium fermion
dynamics in a chiral quark-meson model.
The time evolution for the fermions is described by first-order (integro-)differential
equations for F and ρ : Eqs. (164)–(167). As pointed out above, the initial fermion
spectral function is completely determined by the equal-time anticommutation relation
of fermionic field operators (cf. Eq. (124)). To uniquely specify the time evolution for F
we have to set the initial conditions. The most general (Gaussian) initial conditions for F
respecting spatial homogeneity, isotropy, parity, charge conjugation and chiral symmetry
can be written as
1
 f
F V (t,t ′, p)|t=t ′=0 =
 2
 − n0 (p) ,
 (168)
F V 0 (t,t ′, p)|t=t ′=0
 = 0 .
 (169)
f
Here n0 (p) denotes the initial particle number distribution, whose values can range
between 0 and 1.
3.5. Nonequilibrium dynamics from the 2PI loop expansion
In Sec. 3.4.3 we have derived coupled evolution equations (138) for the statistical
propagator F and the spectral function ρ . A systematic approximation to the exact
equations can be obtained from the loop or coupling expansion of the 2PI effective
action, as discussed in Sec. 2.1. This determines all the required self-energies using (25)
and the decomposition identities (116) and (120) for scalars, and (121) for fermions.
(Approximations for gauge field theories will be discussed in Sec. 6 in the context of
nPI effective actions with n > 2.) We emphasize that all classifications of contributions
are done for the effective action. Once an approximation order is specified on the level
of the effective action, there are no further classifications on the level of the evolution
Introduction to Nonequilibrium Quantum Field Theory
 47
equations. This is a crucial aspect, which ensures the “conserving” properties of 2PI
expansions (cf. the discussion in Sec. 1.1). The purpose of this section is to present the
relevant formulae, whose physics will be explained in detail in the later sections.
We consider first the case of the scalar O(N) symmetric field theory with classical
action (13) and a vanishing field expectation value. The case φ 6= 0 is treated in Sec. 3.6.1
below. From the 2PI effective action to three-loop order (36) one finds with F ab (x, y) =
F(x, y)δab and ρab (x, y) = ρ (x, y)δab the two-loop self-energies:
N + 2
Σ(0) (x) = λ
 F(x, x),
 (170)
6N
Σρ (x, y) = −λ
 2 N 6N + 2
 2
 ρ (x, y) 
F 2
 (x, y) − 12
 1 ρ 2
 (x, y)
 ,
 (171)
ΣF (x, y) = −λ
 2 N 18N + 2
 2
 F(x, y) 
F 2
 (x, y) − 4
 3 ρ 2
 (x, y)
 ,
 (172)
which enter (119) and (138). The two-loop contribution to the effective action adds only
to Σ(0) and corresponds to a space-time dependent mass shift in the evolution equations.
As is discussed in detail below in Sec. 4.1, the time evolution obtained from the two-loop
2PI effective action (Hartree, or similarly leading-order large-N approximations) suffers
from the presence of an infinite number of spurious conserved quantities, which are not
present in the fully interacting theory. As an important consequence no thermalization
can be observed to that order. A crucial ingredient for the description of nonequilibrium
evolution comes from the three-loop contribution to the 2PI effective action as described
by (171) and (172).
For the chiral Yukawa model for N f = 2 fermion flavors and N 2 f scalar fields with
classical action (68) as described in Sec. 2.3.1, one obtains from the two-loop 2PI
effective action the fermion self-energies entering Eqs. (164)–(167):
AV μ
 (x0 , y 0 ; p)
 = −h
2
 μ
 Z
 (2π d3q )3 hF V
 μ (x0 , y0
 ; q) ρ (x0 , y0; p − q)
+ ρ V (x0 , y0; q) F(x0 , y0 ; p − q)i
 ,
 (173)
C V μ
 (x0 , y 0 ; p) = −h2
 Z
 (2π d3q )3 hF V
 μ (x0 , y0
 ; q) F(x0 , y0; p − q)
− 4
 1 ρ V μ
 (x0 , y0; q) ρ (x0 , y0; p − q)i
 .
 (174)
Here ρ V μ
 (x0 , y0 ; q) and F V μ
 (x0, y0 ; q) are the vector components of the fermion two-point
functions as introduced in Sec. 3.4.5. The dynamics of the scalar two-point functions
ρ (x0, y0 ; p) and F(x0 , y0 ; p) is described by the evolution equations (138) with the scalar
self-energies to this loop order:
Σ(0)
 (x) = λ
 N 6N2 2 f + f
 2 Z
 (2π d3q
 )3
 F(x0, x0 ; q),
 (175)
Introduction to Nonequilibrium Quantum Field Theory
 48
Σρ (x0 , y0 ; p) = −
 8h2
 N f
 Z
 (2π d3q )3 ρ V
 μ (x0 , y0
 ; q) F V, μ (x0 , y0; p − q) ,
ΣF (x0 , y0
 ; p) = −
 4h2
 N f
 Z
 (2π d3q )3 hF V
 μ (x0 , y0
; q) F V, μ (x0 , y0; p − q)
− 1 4
 ρ V μ
 (x0 , y0 ; q) ρ V, μ (x0 , y0; p − q)i
 .
3.6. Nonequilibrium dynamics from the 2PI 1/N expansion
(176)
(177)
We consider first the case of the scalar O(N) symmetric field theory with classical
action (13) and a vanishing field expectation value such that F ab (x, y) = F(x, y) δab and
ρab (x, y) = ρ (x, y)δ ab . The case φ 6= 0 is treated in Sec. 3.6.1. In the 1/N–expansion of
the 2PI effective action to next-to-leading order, as discussed in Sec. 2.4, the effective
mass term M 2 (x; G) appearing in the evolution equations (138) is given by
2 N + 2
M 2(x; F) = m + λ
 F(x, x) .
 (178)
6N
One observes that this local self-energy part receives LO and NLO contributions. In
contrast, the non-local part of the self-energy (118) is nonvanishing only at NLO:
Σ(x, y; G) = − λ /(3N) G(x, y)I(x, y) and using the decomposition identities (116) and
(120) one finds
ΣF (x, y) = −
 3N
 λ 
F(x, y)IF (x, y) − 1
 4
 ρ (x, y)I ρ (x, y)
 ,
 (179)
Σρ (x, y) = −
 3N
 λ 
F(x, y)Iρ (x, y) + ρ (x, y)I F (x, y)
 .
 (180)
Here the summation function (84) reads in terms of its statistical and spectral compo-
nents:10
IF (x, y) =
 x0
 λ 6
 F 2
 (x, y) − 1
 4
 ρ 2 (x, y)

 y
0
−
 λ
 6
 ( Z 0
 dz I ρ (x, z)
F 2(z, y) − 4
 1
 ρ 2 (z, y)
 − 2Z 0
 dz I F (x, z)F(z, y)ρ (z, y) )
 , (181)
x
0
Iρ (x, y) =
 λ
 3
 F(x, y)ρ (x, y) −
 λ
 3
 Z dz Iρ (x, z)F(z, y)ρ (z, y) ,
 (182)
y0
10
 i
 This follows from 0
 0
 using the decomposition identity for the propagator (116) and I(x, y) = I F (x, y) −
2 Iρ (x, y) signC (x − y ).
 Cf. also the detailed discussion in Sec. 3.4.3.
Introduction to Nonequilibrium Quantum Field Theory
 49
ΣF using (x, y), the Σρ abbreviated (x, y), IF (x, y) notation and Iρ (x, R
tt 1 y) 2 dz are ≡ real R
tt 1 2 functions.
 dz0 R
−∞
 ∞ dd
 z. Note that F(x, y), ρ (x, y),
For the Yukawa model of Sec. 2.3.1 the results from the 1/N f expansion of the
2PI effective action to NLO can be directly inferred from (173)–(177). Recall that all
classifications are done on the level of the effective action, as explained for fermions at
the end of Sec. 2.4.1. The only difference between the NLO evolution equations and the
result (173)–(177) from the two-loop effective action is that for the former the NNLO
part ∼ N −2
 f in (175) is dropped.
3.6.1. Nonvanishing field expectation value
We consider the scalar N-component field theory with classical action (13). In the
presence of a nonzero field expectation value φa the most general propagator can no
longer be evaluated for the diagonal configuration (35). Restoring all field indices the
evolution equations (138) read

 x δac + Mac
 2
 (x)

 ρcb (x, y) = −
 Zy0
 x0
 x0
 dz Σρ ac (x, z)ρcb(z, y) ,

xδac + Mac
 2
 (x)
 F cb (x, y) = −
 Z
0
y0
 dz Σρ ac (x, z)F cb(z, y)
+
 Z0
 dz ΣFac (x, z)ρ cb(z, y) .
 (183)
The statistical propagator and spectral function components have the properties
F ab
 ∗ (x, y) = F ab
 (x, y) = F 2ba
 (y, x) and 2 ρ ab
 ∗ (x, y) = ρab
 (x, y) = −ρba
 (y, x). At NLO in the
2PI 1/N expansion Mab
 (x) ≡ Mab
 ρ
 (x; φ , F) is ρ
 given by (146) and the self-energies
ΣFab (x, y) ≡ ΣFab (x, y; φ , ρ , F) and Σab (x, y) ≡ Σab (x, y; φ , ρ , F) are obtained from (83) as
ΣFab (x, y)
 = −
 3N
 λ n
IF (x, y) [φa (x)φb (y) +1
 F ab (x, y)] − 4
 1
 Iρ (x, y)ρab (x, y)
ρ
 λ +P F (x, y)F ab (x, y) − 4
 P ρ (x, y)ρab(x, y)o
,
 (184)
Σab (x, y) = −
 3N
 n
Iρ (x, y) [φa(x)φb (y) + F ab (x, y)] + I F (x, y)ρab (x, y)
+P ρ (x, y)F ab(x, y) + P F (x, y)ρab(x, y)o
.
 (185)
The functions IF (x, y) ≡ IF (x, y; ρ , F) and Iρ (x, y) ≡ Iρ (x, y; ρ , F) satisfy the same equa-
tions as for the case of a vanishing macroscopic field given by (181) and (182). The
respective φ -dependent summation functions P F (x, y) ≡ P F (x, y; φ , ρ , F) and P ρ (x, y) ≡
P ρ (x, y; φ , ρ , F) are given by
P F (x, y) = −
 3N
 λ
 (
H F (x, y) −
 Z0
x0 dz 
H ρ (x, z)I F (z, y) + I ρ (x, z)HF (z, y)

Introduction to Nonequilibrium Quantum Field Theory
 50
+
 Z0
x0
 y0
 dz 
 H y0
 F (x, z)Iρ (z, y) + I F (x, z)H ρ (z, y)

−
 Z0
 dz
 Z0
 dv Iρ (x, z)HF (z, v)Iρ (v, y)
+
 Z0
x0
 dz
 Z0
 z0
 dv I ρ (x, z)H ρ (z, v)I F (v, y)
+
 Z0
y0
 dz
 Zz0
 y0
 dv IF (x, z)Hρ (z, v)Iρ (v, y))
,
 (186)
P ρ (x, y) = −
 3N
 λ
 0 (
Hρ (x, 0
 y) −
 Zy0
 x0 dz 
H ρ (x, z)I ρ (z, y) + I ρ (x, z)H ρ (z, y)

+
 Zy0
 x
 dz
 Zy0
 z
 dv I ρ (x, z)Hρ (z, v)I ρ (v, y) )
,
 (187)
with
HF (x, y) ≡ −φa (x)F ab (x, y)φb (y) ,
 Hρ (x, y) ≡ −φa (x)ρab (x, y)φb (y) .
 (188)
The time evolution equation for the field (147) for the 2PI effective action to NLO (83)
is given by

x +
 6N
 λ φ 2
 (x)
 δab + Mab 2
 (x; φ = 0, F)
 φb (x)
=
 3N λ Z x0
 0
 x
0
 dy ρ
 
I ρ (x, y)F ab (x, y) + IF (x, y)ρab (x, y)
 φb (y)
= −
 Z0
 dy Σab (x, y; φ = 0, F, ρ ) φb (y) .
 (189)
The second equality follows from (185).
3.7. Numerical implementation
Beyond the leading-order approximation, the time evolution equations (138) and
(147) are nonlinear integro-differential equations. The approximate self-energies ob-
tained from a loop expansion or 1/N expansion of the 2PI effective action are described
in Secs. 3.5 and 3.6. Though these equations are in general too complicated to be solved
analytically11 without additional approximations, they can be very efficiently imple-
mented and solved on a computer. Here it is important to note that all equations are
11
 Cf. Sec. 4.3 for an approximate analytical solution in the context of parametric resonance.
Introduction to Nonequilibrium Quantum Field Theory
51
explicit in time, i.e. all quantities at some later time t f can be obtained by integration
over the explicitly known functions for times t < t f for given initial conditions. This is
a direct consequence of causality. In this respect, solving the nonequilibrium evolution
equations can be technically simpler than solving the corresponding theory in thermal
equilibrium. The reason is that for the study of thermal equilibrium the equation of the
form (111) is typically employed, which has to be solved self-consistently since on its
l.h.s. and r.h.s. the same variables appear. It involves the propagator for the full range of
its arguments. In contrast, the time-evolution equations (138) for ρ (x, y)|x0 =t1 ,y0 =t2 and
F(x, y)|x 0 =t1 ,y0 =t2 do not depend on the r.h.s. on ρ (x, y) and F(x, y) for times x0 ≥ t1 and
y0 ≥ t2 . To see this note that the integrands vanish identically for the upper time-limits
of the memory integrals because of the anti-symmetry of the spectral components, with
ρ (x, y)|x0 =y0 ≡ 0 and Σρ (x, y)| x0 =y0 ≡ 0. As a consequence, only explicitly known quan-
tities at earlier times determine the time evolution of the unknowns at later times. The
numerical implementation therefore only involves sums over known functions.
To be more explicit we consider first a scalar field theory. For spatially homogeneous
fields it is sufficient to implement the equations for the Fourier components F(t,t ′; p)
and ρ (t,t ′; p) and we consider φ = 0. The numerical implementation with φ 6= 0 follows
along the very same lines for the equation (147). Spatially inhomogeneous fields pose
no complication in principle but are computationally more expensive. As an example
we consider a time discretization t = nat , t ′ = mat with stepsize at such that F(t,t ′) 7→
F(n, m), and
∂ t 2 F(t,t ′) 7→
 at 1 2
 
F(n
 +
 1,
 m)
 +
 F(n
 −
 1,
 m)
 −
 2F(n,
 m)

,
 (190)
t
Z 0
 dtF(t,t ′) 7→ a t 
F(0, m)/2 + n−1
 l=1
 ∑ F(l, m) + F(n, m)/2
 ,
 (191)
where we have suppressed the momentum labels in the notation. The second derivative is
replaced by a finite-difference expression, which is symmetric in at ↔ −at . It is obtained
from employing subsequently the lattice “forward derivative” [F(n +1, m) −F(n, m)]/at
and “backward derivative” [F(n, m) − F(n − 1, m)]/at . The integral is approximated
using the trapezoidal rule with the average function value [F(n, m) + F(n + 1, m)]/2 in
an interval of length a t . The above simple discretization leads already to stable numerics
for small enough stepsize at , but the convergence properties may be easily improved
with more sophisticated standard estimators if required.
As for the continuum the propagators obey the symmetry properties F(n, m) =
F(m, n) and ρ (n, m) = −ρ (m, n). Consequently, only “half” of the (n, m)–matrices have
to be computed and ρ (n, n) ≡ 0. Similarly, since the self-energy Σρ and the summation
function Iρ appearing in the 1/N expansion to NLO (cf. Eq. (182)) are antisymmet-
ric in time one can exploit that Σρ (n, n) and Iρ (n, n) vanish identically. For the case of
the NLO time evolution the equations (138) with (179)–(182) are used to advance the
matrices F(n, m) and ρ (n, m) stepwise in the “n-direction” for each given m. As ini-
tial conditions one has to specify F(0, 0; p), F(1, 0; p) and F(1, 1; p), while ρ (0, 0; p),
ρ (1, 0; p) and ρ (1, 1; p) are fixed by the equal-time commutation relations (117). The
Introduction to Nonequilibrium Quantum Field Theory
 52
time discretized versions of (138) read:
F(n + 1, m; p) = 2F(n, m; p) − F(n − 1, m; p)
−at 2
 
p2
 + m2
 + λ
 N 6N + 2
 Z
k
 F(n, n; k)
 F(n, m; p)
−at 3 (
Σρ (n, 0; p) F(0, m; p)/2 − ΣF (n, 0; p) ρ (0, m; p)/2
 (192)
+
 m−1 l=1
 ∑
 
Σρ (n, l; p) F(l, m; p) − ΣF (n, l; p) ρ (l, m; p)

+ n−1
 l=m
 ∑ Σρ (n, l; p) F(l, m; p))
 ,
(n ≥ m) 12 and similarly for ρ . These equations are explicit in time: Starting with n = 1,
for the time step n + 1 one computes successively all entries with m = 0, . . ., n + 1 from
known functions at earlier times. At first sight this property is less obvious for the non-
derivative expressions (181) for IF and (182) for Iρ whose form is reminiscent of a gap
equation. However, the discretized equation for I ρ ,
I ρ (n, m; q) =
 λ
 3 Z
k
 (
F(n, m; q − k)ρ (n, m; k)
−at
 l=m+1
 n−1
 ∑
 Iρ (n, l; q)F(l, m; q − k)ρ (l, m; k))
 ,
 (193)
shows that all expressions for Iρ (n, m) are explicit as well: Starting with m = n where
I ρ vanishes one should lower m = n, . . . , 0 successively. For m = n − 1 one obtains an
explicit expression in terms of F(n, m) and ρ (n, m) known from the previous time step
in n. For m = n − 2 the r.h.s. then depends on the known function Iρ (n, n − 1) and so on.
Similarly, for given Iρ (n, m) it is easy to convince oneself that the discretized Eq. (182)
specifies I F (n, m) completely in terms of IF (n, 0), . . ., IF (n, m − 1), which constitutes an
explicit set of equations by increasing m successively from zero to n.
It is crucial for an efficient numerical implementation that each step forward in time
does not involve the solution of a self consistent or gap equation. This is manifest in
the above discretization. The main numerical limitation of the approach is set by the
time integrals (“memory integrals”) which grow with time and therefore slow down the
numerical evaluation. Typically, the influence of early times on the late time behavior
is suppressed and can be neglected numerically in a controlled way. In this case, it is
often sufficient to only take into account the contributions from the memory integrals
12 For the discretization of the time integrals it is useful to distinguish the cases n ≥ m and n ≤ m. We
compute the entries F(n + 1, m) from the discretized equations for n ≥ m except for n + 1 = m where we
have to use the equations for n ≤ m.
Introduction to Nonequilibrium Quantum Field Theory
 53
for times much larger then the characteristic inverse damping rate (cf. Sec. 4.1.4 below).
An error estimate then involves a series of runs with increasing memory time.
For scalars one can use a standard lattice discretization for a spatial volume with
periodic boundary conditions. For a spatial volume V = (N s a)d with lattice spacing a
one finds for the momenta:
p2
 →
 7 i=1 ∑ d
 a
 4
 2 sin
2 
 api
 2
 
 ,
 pi =
 2π Nsni
 a
 ,
 (194)
where ni = 0, . . ., Ns − 1. This can be easily understood from acting with the correspond-
ing finite-difference expression (190) for space-components: ∂x 2e−ipx 7→ e−ipx [eipa +
e−ipa −2]/a2 = −e−ipx 4 sin2(pa/2)/a2. On the lattice there is only a subgroup of the ro-
tation symmetry generated by the permutations of px , py , pz and the reflections px ↔ −px
etc. for d = 3. Exploiting these lattice symmetries reduces the number of independent
lattice sites to (Ns + 1)(Ns + 3)(Ns + 5)/48. We emphasize that the self-energies are cal-
culated in coordinate space, where they are given by products of coordinate-space corre-
lation functions, and then transformed back to momentum space. The coordinate-space
correlation functions are available by fast Fourier transformation routines.
The lattice introduces a momentum cutoff π /a, however, the renormalized quantities
are insensitive to cutoff variations for sufficiently large π /a. We emphasize here again
that it is often convenient to carry out the numerical calculations using unrenormalized
equations, or equations where only the dominant (quadratically) divergent contributions
in the presence of scalars are subtracted. Cf. the discussion at the end of Sec. 2.2. In
order to study the infinite volume limit one has to remove finite size effects. Here this is
done by increasing the volume until convergence of the results is observed.13 We finally
note that the nonequilibrium equations are very suitable for parallel computing on PC
clusters using the MPI standard.
For fermionic field theories the numerical implementation is more involved than for
purely scalar theories. Consider the evolution equations for the Yukawa model (164)–
(167), together with the self-energies (173)–(174). The structure of the fermionic equa-
tions is reminiscent of the form of classical canonical equations. In this analogy, F V (t,t ′)
plays the role of the canonical coordinate and F V 0 (t,t ′) is analogous to the canonical mo-
mentum. This suggests to discretize F V (t,t ′ ) and ρ V (t,t ′) at t − t ′ = 2nat (even) and
13
For time evolution problems the volume which is necessary to reach the infinite volume limit to a
given accuracy can depend on the time scale. This is, in particular, due to the fact that finite systems
can show characteristic recurrence times after which an initial effective damping of oscillations can be
reversed. The observed damping can be viewed as the result of a superposition of oscillatory functions
with differing phases or frequencies. The recurrence time is given by the time after which the phase
information contained in the initial oscillations is recovered. Then the damping starts again until twice
the recurrence time is reached and so on. In the LO approximation one can explicitly verify that the
observed recurrence times e.g. for the correlation F(x, x) scales with the volume or the number of lattice
sites to “infinity”. We emphasize that the phenomenon of complete recurrences, repeating the full initial
oscillation pattern after some characteristic time, is not observed once scattering is taken into account.
Periodic recurrences can occur with smaller amplitudes as time proceeds and are effectively suppressed
in the large-time limit.
Introduction to Nonequilibrium Quantum Field Theory
 54
F V 0 (t,t ′) and ρV 0 (t,t ′) at t − t ′ = (2n + 1)a t (odd) time-like lattice sites with spacing at .
This is a generalization of the “leap-frog” prescription for temporally inhomogeneous
two-point functions. This implies in particular that the discretization in the time direc-
tion is coarser for the fermionic two-point functions than for the bosonic ones. This
“leap-frog” prescription may be easily extended to the memory integrals on the r.h.s. of
Eqs. (164)–(167) as well.
We emphasize that the discretization does not suffer from the problem of so-called
fermion doublers. The spatial doublers do not appear since (164)–(167) are effectively
second order in ~ x-space. Writing the equations for ~ F V (t,t ′,~ x) and ~ ρV (t,t ′,~ x) starting
from (164)–(167) one realizes that instead of first order spatial derivatives there is a
Laplacian appearing. Hence we have the same Brillouin zone for the fermions and
scalars. Moreover, time-like doublers are easily avoided by using a sufficiently small
stepsize in time at /as.
The fact that Eqs. (164)–(167) with (173)–(174) and the respective scalar ones contain
memory integrals makes numerical implementations expensive. Within a given numeri-
cal precision it is typically not necessary to keep all the past of the two-point functions
in the memory. A single PIII desktop workstation with 2GB memory allows us to use a
memory array with 470 time-steps (with 2 temporal dimensions: t and t ′). For instance,
we have checked for the presented runs in Sec. 4.2 that a 30% change in the memory
interval length did not alter the results. For a typical run 1-2 CPU-days were neces-
sary. Much shorter times can be achieved with parallel computing on PC clusters using
the MPI standard. The shown plots in that section are calculated on a 470 × 470 × 323
lattice. (The dimensions refer to the t and t ′ memory arrays and the momentum-space
discretization, respectively.) By exploiting the spatial symmetries described in Sec. 3.4.5
the memory need could be reduced by a factor of 30. We have checked that the infrared
cutoff is well below any other mass scales and that the UV cutoff is greater than the
mass scales at least by a factor of three. In the evolution equations we analytically sub-
tract only the respective quadratically divergent terms. To extract physical quantities we
follow the time evolution of the system for a given lattice cutoff and measure e.g. the
renormalized scalar mass from the oscillation frequency of the correlator zero-modes to
set the scale.
4. NONEQUILIBRIUM PHENOMENA
Our aim in this section is to study quantum field theories which capture important as-
pects of nonequilibrium physics and which are simple enough that one can perform a
precise quantitative treatment. We employ first the two-particle irreducible 1/N expan-
sion for the scalar N-component field theory (13). Considering subsequent orders in the
expansion corresponds to include more and more aspects of the dynamics. This allows
one to study the influence of characteristic ingredients such as scattering, off-shell and
memory effects on the time evolution. We note that time-reflection invariance for the
Introduction to Nonequilibrium Quantum Field Theory
 55
equations and conservation of energy is preserved at each order in the 1/N–expansion.14
It should be stressed that during the nonequilibrium time evolution there is no loss of
information in any strict sense. Here we consider closed systems without coupling to an
external heat bath or external fields. There is no course graining or averaging involved in
the dynamics. The important process of thermalization is a nontrivial question in a cal-
culation from first principles. Thermal equilibrium itself is time-translation invariant and
cannot be reached from a nonequilibrium evolution on a fundamental level. It is strik-
ing that we will observe below that scattering drives the evolution very closely towards
thermal equilibrium results without ever deviating from them for accessible times.
Here we study the time evolution for various initial condition scenarios away from
equilibrium. One corresponds to a “quench” often employed to mimic the situation of
a rapidly expanding hot initial state which cools on time scales much smaller than the
relaxation time of the fields. Initially at high temperature we consider the relaxation
processes following an instant “cooling” described by a sudden drop in the effective
mass. Another scenario is characterized by initially densely populated modes in a narrow
momentum range around ±pts. This so-called “tsunami” initial condition is reminiscent
of colliding wave packets moving with opposite and equal momentum. A similar non-
thermal and radially symmetric distribution of highly populated modes may also be
encountered in the so-called “color glass” state at saturated gluon density. Of course,
a sudden change in the two-point function of a previously equilibrated system or a
peaked initial particle number distribution are general enough to exhibit characteristic
properties of nonequilibrium dynamics for a large variety of physical situations. We
will first consider results for the 1 + 1 dimensional quantum field theory, since technical
aspects such as the infinite volume limit, renormalization and large times can be all
implemented with great rigour. The study also has the advantage that the physics of off-
shell effects can be very clearly discussed since they play a more important role for one
spatial dimension as compared to three dimensions. The conclusions we will draw are
not specific to scalar theories in low dimensions, and we will compare below with the
corresponding results in 3 + 1 dimensions for scalars interacting with fermions.
For the 2PI 1/N expansion we will first consider the time evolution at leading order
(LO) and compare it with the next-to-leading order dynamics (NLO). The “mean-field
type” LO approximation has a long history in the literature. However, a drastic con-
sequence of this approximation is the appearance of an infinite number of conserved
quantities, which are not present in the interacting finite-N theory. These additional con-
stants of motion can have a substantial impact on the time evolution, since they strongly
constrain the allowed dynamics. As an important consequence of the LO approximation,
the late-time behavior depends explicitly on the details of the initial conditions and the
approach to thermal equilibrium cannot be observed in this case as is shown below. The
extensive use of this approximation was based on the hope that deviations from the LO
behavior are not too sizeable for not too late times. By taking into account the NLO
contributions it turns out, however, that for many important questions there are substan-
tial corrections long before thermalization sets in. In contrast, we will find that various
approximations of the 2PI effective action which go beyond “mean-field type” (LO,
14
 These properties are also taken care of by the employed numerical techniques.
Introduction to Nonequilibrium Quantum Field Theory
56
Hartree or any Gaussian) dynamics show thermalization and give comparable answers.
The 2PI approximations beyond mean-field share the property that the spurious con-
stants of motion are absent, which emphasizes the important role of conservation laws
for the nonequilibrium dynamics: fake conserved quantities keep the information about
the initial conditions for all times and can spoil any effective loss of details about the
early-time behavior.
4.1. Scattering, off-shell and memory effects
4.1.1. LO fixed points
For simplicity we consider spatially homogeneous field expectation values φa(t) =
hΦa (t, x)i, such that we can use the Fourier modes F ab (t,t ′; p) and ρab(t,t ′ ; p).15 The so-
lution of (138) in the limit of a free field theory (λ ≡ 0) describes modes which oscillate
less with they frequency are notp identically p2 + m2 /2π zero. for Equal-time unequal-time correlation functions modes F(t,F(t,t; 0; p) and p) oscillate ρ (t, 0; p), either
 un-
with twice that frequency or they are constant in time (ρ (t,t; p) ≡ 0 according to Eq.
(117)). The latter corresponds to solutions which are translationally invariant in time,
i.e. F(t,t ′; p) = F(t − t ′ ; p) and ρ (t,t ′ ; p) = ρ (t − t ′ ; p).
The LO contribution to the 2PI effective action (78) adds a time-dependent mass shift
to the free field evolution equation. The resulting effective mass term, given by (146)
for N → ∞, is the same for all Fourier modes and consequently each mode propagates
“collisionlessly”. There are no further corrections since according to (79) and (83) the
self-energies ΣF and Σρ are O(1/N) and vanish in this limit! The evolution equations
(138) for this approximation read:
∂t ∂t 2
 2
 + + p2 p2 + + M M2 2(t; (t; φ φ , ,F) F) 
 ρab F ab (t,t (t,t ′; ′; p) p) = = 0 0 ,
 ,
 (195)
∂t 2 +
  λ φ 2
 (t) + M2(t; φ 
 0, F) φb (t) = 0 ,

 6N
 ≡ 
with
M 2
 (t; φ , F) ≡ m2
 +
 6N
 λ
 Z
p
 F cc(t,t; p) + φ 2
(t)
 ,
 (196)
completely where R
p ≡ decoupled R
 dd p/(2π from )d . In ρ this . Similar case to one the observes free field that theory the evolution limit, to LO of the F and spectral
 φ is
function does not influence the time evolution of the statistical propagator. The reason is
that in this
 approximation the spectrum consists only of “quasiparticle” modes of energy
ωp (t) = pp2 + M 2(t) with an infinite life-time. The associated mode particle numbers
15
 Note that a spatially homogeneous field expectation value contains all fluctuations arising from an
inhomogeneous Φ(x) as well. From a practical point of view the extraction of physics related to inhomo-
geneous field fluctuations is, of course, more difficult since it is contained in higher correlation functions.
Introduction to Nonequilibrium Quantum Field Theory
 57
are conserved for each momentum separately. In contrast, in the interacting quantum
field theory particles decay, get created, annihilated and the notion of a conserved
particle number is clearly absent for real scalar fields. Note that there are also no memory
integrals appearing on the r.h.s. of (195), which incorporate in particular direct scattering
processes.
As an example we consider a “tsunami” type initial condition in 1 + 1 dimension. The
initial statistical propagator is given by
n p(0) + 1/2
F(0, 0; p) = ,
 ∂ t F(t, 0; p)| t=0 = 0 ,
F(0, 0; p)∂t ∂t ′ p
 F(t,t p2 + ′; M p)| 2 t=t (0)
 ′=0 = [n p(0) + 1/2]2 ,
 (197)
φ (0) = ∂t φ (t)|t=0 = 0 .
 (198)
Here F ab (t,t ′; p) ≡ 2 F(t,t ′; p)δab , which is valid for all times with these initial conditions.
The mass term M (0) is given by the gap equation (196) in the presence of the initial
non-thermal particle number distribution
n p(0) = A e
−
 2σ 1 2
 (|p|−pts
)2
 .
 (199)
As the renormalization condition we choose the initial renormalized mass in vacuum,
mR ≡ M(0)| n(0)=0 = 1, as our dimensionful scale. There is no corresponding coupling
renormalization in 1 + 1 dimension. In these units the particle number is peaked around
|p| = pts = 5mR with a width determined by σ = 0.5mR and amplitude A = 10. We
consider the effective coupling λ /(6m2 R ) = 1.
On the left of Fig. 3 we present the time evolution of the equal-time correlation
modes F(t,t; p) for different momenta: zero momentum, a momentum close to the
maximally populated momentum pts and about twice pts . One observes that the equal-
time correlations at LO are strictly constant in time. This behavior can be understood
from the fact that for the employed “tsunami” initial condition the evolution starts
at a time-translation invariant non-thermal solution of the LO equations. There is an
infinite number of so-called fixed point solutions which are constant in time. If the real
world would be well approximated by the LO dynamics then this would have dramatic
consequences. The thermal equilibrium solution, which is obtained for an initial Bose-
Einstein distribution instead of (199), is not at all particular in this case! Therefore,
everything depends on the chosen initial condition details. This is in sharp contrast to the
well-founded expectation that the late-time behavior may be well described by thermal
equilibrium physics. Scattering effects included in the NLO approximation indeed drive
the evolution away from the LO non-thermal fixed points towards thermal equilibrium,
which is discussed in detail in Sec. 4.1.2 below.
A remaining question is what happens at LO if the time evolution does not start
from a LO fixed point, as was the case for the “tsunami” initial condition above. In
the left graph of Fig. 4 we plot the evolution of M2 (t) in the LO approximation as a
function of time t, following a “quench” described by an instant drop in the effective
mass term
 from 2M2
(0) to M 2
 (0). The initial particle number distribution is n p(0) =
Introduction 1/(exp[p p2 to Nonequilibrium + 2M 2(0)/T 0 ] Quantum − 1) with Field T Theory
 0 = 2M(0) and φ (0) = φ̇ (0) = 0. The sudden
 58
1.4
]
 (NLO)
) 1.2
 t=0
2
 t (LO
 pn)
 / 1
p NLO
 1; t +, p=0
t 1 0.8
( p=0.94 pts
 [F 1
 p=1.76 pts
 g o0.6
L ∆t=10
NLO
LO
 0.4
NLO
0
 LO
 0.2
0
 200
 t
 400
 600
 M(0) 2.5
 3
 3.5
 εp
 4
 4.5
 5
FIGURE 3. LEFT: Comparison of the LO and NLO time dependence of the equal-time correlation
modes F(t,t; p) for the “tsunami” initial condition (197)–(199). The importance of scattering included in
the NLO approximation is apparent: the non-thermal LO fixed points become unstable and the “tsunami”
decays, approaching thermal equilibrium at late times. RIGHT: Effective particle number distribution for
a “tsunami” in the presence of a thermal background. The solid line shows the initial distribution which
for low and for high momenta follows a Bose-Einstein distribution, i.e. Log[1 + 1/n p(0)] ≃ εp(0)/T 0.
At late times the non-thermal distribution equilibrates and approaches a straight line with inverse slope
T eq > T 0 .
change in the effective mass term drives the system out of equilibrium and one can
study its relaxation. We present M 2 (t) for three different couplings λ = λ 0 ≡ 0.5 M2(0)
(bottom), λ = 10λ0 (middle) and λ = 40λ0 (top). All quantities are given in units of
appropriate powers of the initial-time mass M(0). Therefore, all curves in the left graph
of Fig. 4 start at one. The time-dependent mass squared M2(t) shoots up in response to
the “quench” and stays below the value 2M(0)2 of the initial thermal distribution. The
amplitude of initial oscillations is quickly reduced and, averaged over the oscillation
time-scale, the evolution is rapidly driven towards the asymptotic values.
We compare the asymptotic values with the self-consistent solution of the LO mass
equation (196) for constant mass squared Mgap
 2 and given particle number distribution
n p (0):16
Mgap 2
 = m2
 +
 λ 6 Z
 2π
 dp
 
n p(0) +
 1
 2
 
 p2 + 1
 Mgap
 2
 .
 (200)
2 q
The result from this gap equation is Mgap
 = {1.01, 1.10, 1.29}M 2(0) for the three values
of λ , respectively. For this wide range of couplings the values are in good numerical
agreement with the corresponding dynamical large-time results inferred from Fig. 4 as
{1.01, 1.11, 1.31}M 2(0) at t = 200/M(0). We explicitly checked that at t = 400/M(0)
16
 Here the logarithmic divergence of the one-dimensional integral is absorbed into the bare mass param-
eter m2 using the same renormalization
 condition as for the dynamical evolution in the LO approximation,
i.e. m2
 + λ 6 R 2π dp
 n p (0) + 1
 2  (p2
 + M 2
(0))−1/2 = M 2 (0).
Introduction to Nonequilibrium Quantum Field Theory
 59
1.5
1.4
)
 1.3
t(2M1.2
(LO)
6
2
M2(x)
t
)
 1
0 0
 5
 10
= 5
p;t,t(F4
1.1
NLO
+
LO3
1
0
 100
 200
 0
 20
 40
 60
t
 t
FIGURE 4. LEFT: Shown is the time-dependent mass term M2 (t) in the LO approximation for three
different couplings following a “quench”. All quantities are given in units of appropriate powers of the
initial-time mass M(0). RIGHT: Time dependence of the equal-time zero-mode F(t,t; p = 0) after a
“quench” (see text for details). The inset shows the mass term M 2 (t), which includes a sum over all
modes. The dotted lines represent the Hartree approximation (LO+ ), while the solid lines give the NLO
results. The coupling is λ /6N = 0.17 M2 (0) for N = 4.
these values are still the same. One concludes that the asymptotic behavior at LO is well
described in terms of the initial particle number distribution n p(0). The latter is not a
thermal distribution for the late-time mass terms with values smaller than 2M2(0).
The LO approximation for F(t,t ′ ; p) and ρ (t,t ′; p) becomes exact17 for t,t ′ → 0, since
the memory integrals on the r.h.s. of the exact Eqs. (138) vanish at initial time. For very
early times one therefore expects the LO approximation to yield a quantitatively valid
description. However, from Fig. 4 one observes that the time evolution is dominated
already at early times by the non-thermal LO fixed points. As discussed above, the
latter are artefacts of the approximation. Though the precise numerical values of the
LO fixed points are pure artefacts, we emphasize that the presence of approximate fixed
points governing the early-time behavior is a qualitative feature that can be observed
also beyond LO (cf. below). The question of how strongly the LO late-time results
deviate from thermal equilibrium depends of course crucially on the details of the initial
conditions. Typically, time- and/or momentum-averaged quantities are better determined
by the LO approximation than quantities characterizing a specific momentum mode.
This is exemplified on the right of Fig. 4, which shows the equal-time zero-mode
F(t,t; p = 0) along with M2(t) including the sum over all modes. Here we employ
a “quench” with a larger drop in the effective mass term from 2.9M 2 (0) to M 2 (0).
T The 0 = initial 8.5M(0). particle In the number figure distribution the dotted is curves n p (0) = show 1/(exp[ the p dynamics p2 + M2(0)/T obtained 0 ] − from 1) with
 an
“improved” LO (Hartree) approximation, LO+ , that takes into account the local part of
the NLO self-energy contribution. The resulting equations have the very same structure
17
 This is due to the fact that we choose to consider Gaussian initial conditions.
Introduction to Nonequilibrium Quantum Field Theory
60
1
)
0=p;0,t(F0
NLO
LO
100
|ρ(t,0;p=0)||F(t,0;p=0)|(NLO)
(NLO)
−1
0
 10
 20
 30
 40
 50
 60
 70
 80
 0
 10
 20
 30
 40
 50
 60
 70
 80
t
 t
FIGURE 5. LEFT: Shown is the evolution of the unequal-time correlation F(t, 0; p = 0) after a
“quench”. Unequal-time correlation functions approach zero in the NLO approximation and correlations
with early times are effectively suppressed (λ /6N = (5/6N ≃ 0.083) M2(0) for N = 10). In contrast, there
is no decay of correlations with earlier times for the LO approximation. RIGHT: The logarithmic plot of
|ρ (t, 0; p = 0)| and |F(t, 0; p = 0)| as a function of time t shows an oscillation envelope which quickly
approaches a straight line. At NLO the correlation modes therefore approach an exponentially damped
behavior. (All in units of M(0).)
as the LO ones, however, with the LO and NLO contribution to the mass term M 2(t)
included as given by Eq. (178) below. The large-time limit of the mass term in the LO+
approximation is determined by the LO+ fixed point solution in complete analogy to the
above discussion. We also give in Fig. 4 the NLO results, which are discussed below.
The effective loss of details about the initial conditions is a prerequisite for thermal-
ization. At LO this is obstructed by an infinite number of spurious conserved quantities
(mode particle numbers), which keep initial-time information. An important quantity
in this context is the unequal-time two-point function F(t, 0; p), which characterizes the
correlations with the initial time. Clearly, if thermal equilibrium is approached then these
correlations should be damped. On the left of Fig. 5 the dotted line shows the unequal-
time zero-mode F(t, 0; p = 0) following the same “quench” at LO as for Fig. 4 left. One
observes no decay of correlations with earlier times for the LO approximation. Scatter-
ing effects entering at NLO are crucial for a sufficient effective loss of memory about
the initial conditions, which is discussed next.
4.1.2. NLO thermalization
In contrast to the LO approximation at NLO the self-energies ΣF and Σρ ∼ O(1/N)
do not vanish. For the initial conditions (197) all correlators are diagonal in field index
space and φ ≡ 0 for all times. In this case the evolution equations derived from the
NLO 2PI effective action (79) are given by (138) with the self-energies (178)–(182).
As discussed in Sec. 3.4.3, the NLO evolution equations are causal equations with
characteristic “memory” integrals, which integrate over the time history of the evolution
Introduction to Nonequilibrium Quantum Field Theory
 61
taken to start at time t0 = 0 without loss of generality.
We consider first the same “tsunami” initial condition (197) as for the LO case
discussed above. The result is shown on the left of Fig. 3 for N = 10. One observes a
dramatic effect of the NLO corrections! They quickly lead to a decay of the initially high
population of modes around pts . On the other hand, low momentum modes get populated
such that thermal equilibrium is approached at late times. In order to make this apparent
one can plot the results in a different way. For this we note that according to (197) the
statistical propagator corresponds to the ratio of the following particle number at initial
time t = 0:
and the corresponding np (t) mode + 2
 1
 energy. = 
F(t,t; Here p)K(t,t; we have p) − defined:
 Q2 (t,t; p)
 1/2
 ,
 (201)
K(t,t ′ ; p) ≡ ∂t ∂ t ′ F(t,t ′; p)
 ,
 Q(t,t ′; p) ≡
 2
 1 
∂t F(t,t ′; p) + ∂t ′ F(t,t ′; p)
 .
 (202)
Since we employed Q(0, 0; p) = 0 for the initial conditions (197) the initial mode energy
is given by
εp(t) =
 
 K(t,t; F(t,t; p)
 p) 
1/2
 ,
 (203)
such that F(t,t; p) = [np(t) + 1/2]/εp(t). For illustration of the results we may use
(201) and (203) for times t > 0 in order to define an effective mode particle number
and energy, where we set Q(t,t ′ ; p) ≡ 0 in (201) for the moment. The behavior of the
effective particle number is illustrated on the right of Fig. 3, where we plot Log(1 +
1/np (t)) as a function of εp(t). Note that for a Bose-Einstein distributed effective particle
number this is proportional to the inverse temperature: Log(1 + 1/nBE (T )) ∼ 1/T . For
the corresponding plot of Fig. 3 we have employed an initial “tsunami” at pts/mR =
2.5, where we added an initial “thermal background” distribution with temperature
T 0/mR = 4.18 Correspondingly, from the solid line (t = 0) in the right figure one observes
the initial “thermal background” as a straight line distorted by the non-thermal “tsunami”
peak. The curves represent snapshots at equidistant time steps ∆t mR = 10. After rapid
changes in n p (t) at early times the subsequent curves converge to a straight line to high
accuracy, with inverse slope T eq /T 0 = 1.175. The initial high occupation number in
a small momentum range decays quickly with time. More and more low momentum
modes get populated and the particle distribution approaches a thermal shape.
The crucial importance of the NLO corrections for the nonequilibrium dynamics can
also be observed for other initial conditions. The right graph of Fig. 4 shows the equal-
time zero-mode F(t,t; p = 0), along with M 2 (t) including the sum over all modes, fol-
lowing a “quench” as described in Sec. 4.1.1. While the dynamics for vanishing self-
energies ΣF and Σρ is quickly dominated by the spurious LO fixed points, this is no
longer the case once the NLO self-energy corrections are included. In particular, one
observes a very efficient damping of oscillations at NLO. This becomes even more pro-
nounced for unequal-time correlators as shown for F(t, 0; p = 0) in the left graph of
18
 The initial mass term is M(0)/mR = 2.24 and λ /6N = 0.5m2 R for N = 4.
Introduction to Nonequilibrium Quantum Field Theory
62
Fig. 5. We find that the unequal-time two-point functions approach zero for the NLO
approximation and correlations with early times are effectively suppressed. Of course,
time-reversal invariance implies that the oscillations can never be damped out to zero
completely during the nonequilibrium time evolution, however, zero can be approached
arbitrarily closely. In the NLO approximation we find that all modes F(t,t ′; p) and
ρ (t,t ′; p) approach an approximately exponential damping behavior for both equal-time
and unequal-time correlations. On the right of Fig. 5 the approach to an exponential be-
havior is demonstrated for |ρ (t, 0; p = 0)| and |F(t, 0; p = 0)| with the same parameters
as for the left figure. The logarithmic plot shows that after a non-exponential period at
early times the envelope of oscillations can be well approximated by a straight line. From
an asymptotic envelope fit of F(t, 0; p = 0) to an exponential form ∼ exp(− γ0
 (damp)
 t) we
obtain here a damping rate γ0
 (damp)
 = 0.016M(0). For comparison, for the parameters
employed for the right graph of Fig. 4 we find γ0
 (damp)
 = 0.11M(0) from an exponential
fit to the asymptotic behavior of F(t, 0; p = 0). The oscillation frequency of F(t, 0; p = 0)
is found to quickly stabilize around 1.1M(0)/2π ≃ 0.18M(0), which is of the same or-
der than the damping rate. Correspondingly, one observes in Fig. 4 an equal-time zero
mode which is effectively damped out at NLO after a few oscillations. The oscillation
frequency is always found to stabilize very quickly and to be quantitatively well de-
scribed by εp(t)/2π for the unequal-time modes (εp(t)/π for the equal-time modes)
with the effective mode energy εp (t) given by (203). We emphasize that the parameter
M(t) given by (178) cannot be used in general to characterize well the oscillation fre-
quencies of the correlation zero-modes. Beyond LO or Hartree the renormalized mass,
which can be obtained from the zero-mode oscillation frequency, receives corrections
from the non-vanishing self-energies ΣF and Σρ .
The strong qualitative difference between LO and NLO appears because an infinite
number of spurious conserved quantities is removed once scattering is taken into ac-
count. It should be emphasized that the step going from LO to NLO is qualitatively very
different than the one going from NLO to NNLO or further. In order to understand better
what happens going from LO to NLO we consider again the effective particle number
(201). It is straightforward by taking the time derivative on both sides of (201) to obtain
an exact evolution equation for np(t) with the help of the exact relations (138):

np (t) +
 1
 2
 
 ∂ t np (t) =
Z t0
 t
dt
 ′′
 n
Σρ (t,t ′′; p)F(t ′′,t; p) − ΣF (t,t ′′; p)ρ (t ′′,t; p)
 ∂t F(t,t ′; p)|t=t ′
 (204)
− 
Σρ (t,t ′′; p)∂ t F(t ′′,t; p) − ΣF (t,t ′′; p)∂t ρ (t ′′,t; p)
 F(t,t; p)
o
Here t0 denotes the initial time which was set to zero in (138) without loss of generality.
Since ΣF ∼ O(1/N) as well as Σρ , one directly observes that at LO, i.e. for N → ∞,
the particle number for each momentum mode is strictly conserved: ∂ t np (t) ≡ 0 at LO.
Stated differently, eq. (201) just specifies the infinite number of additional constants
of motion which appear at LO. In contrast, once corrections beyond LO are taken
Introduction to Nonequilibrium Quantum Field Theory
 63
into account then (201) no longer represent conserved quantities. The first non-zero
contribution to the self-energies occurs at NLO. As a consequence ∂t np(t) 6≡ 0 in general.
For the LO approximation we have seen that there is an infinite number of time-
translation invariant solutions (201) to the nonequilibrium evolution equations. Thermal
equilibrium plays no particular role at LO. However, this is very different at NLO and
beyond. It is very instructive to consider what is required to find solutions of (204) which
are homogeneous in time. Of course, any time-translation invariant solution cannot
be achieved if one respects all symmetries of the quantum field theory such as time
reflection symmetry. However, we can consider the (hom) limit t0 → −∞ and ask under what
conditions time-translation invariant F(t,t ′; p) = F (t −t ′; p) etc. represent solutions
of (204). Using the (anti-)symmetry of the (spectral function) statistical propagator we
have
ρ
 (hom)
 (t − t ′
 ; p) = −i
 Z
 dω
 2π
 sin[ω (t − t ′ )] ρ (hom)(ω , p) ,
F (hom) (t − t ′ ; p) =
 Z
 dω
 2π
 cos[ω (t − t ′ )] F (hom) (ω , p) .
 (205)
Note (hom) that in our conventions (hom) ρ (hom) (ω ; p) = −ρ (hom) (hom) (−ω ; p) is purely imaginary, while
F (ω ; p) = F (−ω ; p) is real. Since ∂ t F (t −t ′ ; p)| t=t ′ ≡ 0, time-translation
invariant solutions of (204) require:
(t−t0 lim
 )→∞ Zt0
 t
dt ′′ h
Σρ
 (hom)
 (t − t ′′; p)∂t F (hom) (t ′′ − t; p)
−ΣF (hom)
 (t − t ′′ ; p)∂t ρ (hom) (t ′′ − t; p)i
 = 0 .
 (206)
Inserting the Fourier transforms (205) and the respective ones for the self-energies and
using
lim
 sin [ω (t − t0)]
 = πδ (ω ) ,
 (207)
(t−t0 )→∞
 ω
one finds that the condition (206) can be equivalently written as
Z
 dω ω h
Σρ
 (hom)
 (ω , p)F
 (hom)
 (ω , p) − ΣF (hom)
 ( ω , p)ρ
 (hom)
(ω , p)i
 = 0 .
 (208)
In contrast to the LO approximation, at NLO and beyond we find with Σ(hom) 6= 0 that
time-translation invariant solutions are highly nontrivial. However, one observes from
the equilibrium fluctuation-dissipation relation (129) with (130) that indeed thermal
equilibrium singles out a solution which fulfills this condition. We emphasize that in
equilibrium the fluctuation-dissipation relation is fulfilled at all orders in the 2PI 1/N
expansion. In this respect there is no qualitative change going from NLO to NNLO or
beyond. In particular, very similar results as shown in Fig. 3 are obtained from the loop
expansion of the 2PI effective action beyond two-loop order, which is demonstrated in
Sec. 4.1.4. Three-loop is required since the two-loop order corresponds to a Gaussian
approximation, which suffers from the same infinite number of spurious conserved
quantities as the LO approximation.
Introduction to Nonequilibrium Quantum Field Theory
 64
We emphasize that the above discussion does not imply that the evolution always
has to approach thermal equilibrium at late times. There are many initial conditions for
which the system cannot thermalize. Obvious examples are initial conditions inferred
from statistical ensembles which do not obey the clustering property. These do not
exhibit thermal correlation functions in general.
4.1.3. Detour: Boltzmann equation
The nonequilibrium 2PI effective action can be employed to obtain effective kinetic
or Boltzmann-type descriptions for quasiparticle number distributions. The latter are
widely used in the literature and it is important to understand their merits and limits.
The “derivation” of such an equation from the 2PI effective action requires a number of
additional approximations which go beyond a loop or 1/N expansion of the 2PI effective
action. At the end of this procedure we will find an irreversible equation, which therefore
lost part of the symmetries of the underlying quantum field theory. The comparison
will allow us, in particular, to clearly point out the role of neglected off-shell effects as
compared to the quantum field theory description.
Kinetic descriptions concentrate on the behavior of particle number distributions.
The latter contain information about the statistical propagator F(x, y) at equal times
x0 = y0 . Correlations between unequal times are outside the scope of a Boltzmann
equation. We can therefore start the discussion from the exact equation for the effective
particle number (204).19 For our current purposes it is irrelevant that we will consider
a spatially homogeneous one-component field theory with vanishing field expectation
value (φ = 0). The steps leading to the Boltzmann equation are summarized as follows:
1. Consider a 2PI loop expansion of Γ[G] to three-loop order. This leads to self-energy
corrections up to two loops, diagrammatically given by
 ,
 , which lead to
the time-dependent mass term
and
ΣF (t,t ′
; p)Σρ (t,t ′
; p)M 2(t) = m2 +
 λ
 2
 Z
p
 F(t,t; p)
 (209)
==−
 λ 6 2
 Z
q,k
 F(t,t ′
; p − q − k)
F(t,t ′
; q)F(t,t ′
; k) − 4
 3
 ρ (t,t ′
; q)ρ (t,t ′
; k)
 ,
−
 λ
 2 2
 Z
q,k
 ρ (t,t ′
; p − q − k)
F(t,t ′
; q)F(t,t ′
; k) − 12
 1
 ρ (t,t ′
 ; q)ρ (t,t ′
 ; k)
 .
(210)
19
 For this it is useful to rewrite the following characteristic term as:
The difference − of i Σρ the Ftwo − ΣF terms ρ 
 = on ΣF the − r.h.s. iΣρ /2 can 
 (F be + directly iρ /2) − interpreted ΣF + iΣρ /2 as 
 the (F difference − iρ /2) .
 of a “loss” and a
“gain” term in a Boltzmann type description.
Introduction to Nonequilibrium Quantum Field Theory
 65
2. Choose a quasiparticle ansatz, i.e. a free-field type form for the two-point functions
and their derivative. This corresponds to the replacements on the r.h.s. of (204):
F(t ρ (t ′′,t; ′′,t; p) p) → → sin (np + ωp 1/2) (t ′′ − cos t) 
/ωp ωp(t ,
 ′′ − t)
 /ωp ,
 (211)
∂t F(t ′′,t; p) (np
 + 1/2) sin 
 ωp (t ′′ t) ,
∂ t ρ (t ′′,t; p) → → − cos ωp (t ′′ − 
t) .
 − 
 (212)
3. Here the particle number np and mode 
 energy ωp 
 may still depend weakly on time:
One assumes a separation of scales, with a sufficiently slow time variation of np (t) such
that one can pull all factors of np(t) out of the time integral on the r.h.s. of (204). The
particle numbers are then evaluated at the latest time of the memory integral.
4. Send the initial time t0 to the remote past: t0 → −∞. Of course, by construction
the resulting equation is not meant to describe the detailed early-time behavior since
t0 → −∞. In the context of kinetic descriptions, one finally specifies the initial condition
for the effective particle number distribution at some finite time and approximates the
evolution by the equation with t 0 in the remote past.
A standard alternative derivation employs first steps 1. and 4., then a first-order
gradient expansion of the two-point functions in the center coordinate (t + t ′)/2 and
then a quasiparticle ansatz. However, for the typically employed first-order gradient
approximation both approaches are fully equivalent. The current procedure has the
advantage that one can send the initial time t0 → −∞ last, which allows one to discuss
a few finite-time effects that are typically discarded. Applying the assumptions 1.–3. to
the exact equation (204) leads after some lengthy shuffling of terms to the expression:
∂t np(t) =
 λ 3
 2
 Z
sqk
 (2π )d δ (p − q − k − s)
 2ωp 2ωq 1
 2ωk 2ωs
(
[(1 + np )(1 + nq )(1 + nk )(1 + ns ) − np nq n kns ]
 (I)
+ Z 3t0
 [(1 t
 dt + ′′ cos np)(1 
( ωp + nq)(1 + ωq + + ωk nk)ns + ωs)(t np nq − nk(1 t ′′)

 + ns )]
 (II)
−+ 3 Zt0
 [(1 t
 dt + ′′ cos np)(1 
( ωp + nq)nkns + ωq + ωk np − nq(1 ωs)(t + − nk t ′′)
 )(1 
 + ns )]
 (III)
−+ [(1 Zt0
 t
 dt + ′′ np cos )nq 
 nk ( ωp ns + ωq np (1 − + ωk nq)(1 − ωs)(t + nk)(1 − t ′′)

 + ns)]
 (IV)
−Zt0
 t
 dt ′′ cos 
( ωp − ωq − ωk − ωs)(t − t ′′)
)
 .
 (213)
Introduction to Nonequilibrium Quantum Field Theory
 66
The contributions on the r.h.s. of this equation have a typical “gain & loss” structure
with a simple interpretation:
(I) describes production and annihilation of four “quasiparticles” (0 → 4, 4 → 0).
(II) and (IV) describe 1 → 3 and 3 → 1 processes.
(III) describes 2 ↔ 2 scattering processes, which are the well-known contributions to
the standard Boltzmann equation.
We emphasize that only the processes described by (III) lead to a non-zero contribution
in the limit (t − t0 ) → ∞. For time-independent mode energies ωp one can perform the
time integrations in (213) explicitly. This leads using (207) to δ -functions, which enforce
the strict conservation of the mode energies involved in the scattering. At this point the
description of the quantum field theory has been fully reduced to the physics of colliding
“on-shell” quasiparticles. Of course, strict energy and momentum conservation for these
quasiparticles leads to vanishing contributions (I), (II) and (IV). The latter describe
processes, which change the total particle number such as 1 → 3 processes. We will
see below that in a quantum field theory “off-shell” particle number changing processes
such as described by (II) can play a very important role! An obvious example is given
by the fact that the Boltzmann equation including the contribution (III) admits grand
canonical thermal solutions with nonzero chemical potential μ :
1
np(t) →
 e(ωp −μ )/T
 1
 .
 (214)
−However, the real scalar quantum field theory is charge neutral and μ ≡ 0! We conclude
that taking into account only 2 ↔ 2 scattering processes of “on-shell” quasiparticles ob-
viously fails to describe the quantum world if for a given initial condition the chemical
potential turns out to be large. We emphasize that total particle number changing pro-
cesses are included in the original quantum field theory equation before applying the
additional steps 2.–4. In particular, they do not only appear as contributions from finite
initial-time effects as discussed here for illustration. The main “off-shell” effects are
removed in step 2., when the quasiparticle assumption enters the description.
The physics of “off-shell” processes can be very nicely observed in the 1 + 1 dimen-
sional quantum field theory. Note that for one spatial dimension the contribution (III)
servation, vanishes as p +q well in the = limit 0 (t − p t0 2
 ) + → M 2
 ∞! + In
 q this 2
 + case M 2
 both
 √k2
 momentum
 + M 2
 √s2 and + M energy 2 = 0, con-
 lead
to an “ineffective” −k two-to-two −s ∧ p scattering where p the incoming − and outcoming − modes are
unaffected: for instance with p = k and q = s the contribution (III) in Eq. (213) vanishes
identically. The equation becomes trivial in this case. We conclude that for the 1 + 1
dimensional quantum field theory there is no dynamics from the Boltzmann equation!
In contrast, from the same three-loop approximation of the 2PI effective action without
the steps 2.–4. we will see in the next section that the 1 + 1 dimensional quantum field
theory thermalizes.
Introduction to Nonequilibrium Quantum Field Theory
 67
F
4.0
)
 3.0
p,ω;0X2.0
(ρ1.0
0.008
0.006
0.004
0.002
 3ω 0
0.000
3
 3.5 4
 4.5 5
 5.5
0.0
0
 1
 ω 0
 2
 ω
 3
 4
 5
 6
FIGURE 6. LEFT: Examples for the time dependence of the equal-time propagator F(t,t; p) with
Fourier modes p = 0, 3, 5 from the 2PI three-loop effective action. The evolution is shown for three
very different nonequilibrium initial conditions with the same energy density (all in initial-mass units).
RIGHT: Wigner transform ρ (X 0; ω , p) of the spectral function as a function of ω at X 0 = 35.1 for p = 0
(in units of mR ). Also shown are fits to a Breit-Wigner function (dotted) with ( ω0 , Γ0) = (1.46, 0.37). The
inset shows a blow-up around 3ω0. The expected bump from “off-shell” 1 ↔ 3 processes is small but
visible.
4.1.4. Characteristic time scales
Nonequilibrium dynamics requires the specification of an initial state. A crucial
question of thermalization is how quickly the nonequilibrium system effectively looses
the details about the initial conditions, and what are the characteristic stages of a partial
loss of information. Thermal equilibrium keeps no memory about the time history except
for the values of a few conserved charges. As a consequence, for the real scalar field
theory thermalization requires that the late-time result is uniquely determined by energy
density.
In Fig. 6 we show the time dependence of the equal-time propagator F(t,t; p) for three
Fourier modes p = 0, 3, 5 and three very different initial conditions with the same energy
density. All quantities are given in appropriate units of the mass at initial time, and we
consider 1 + 1 dimensions. For the solid line the initial conditions are close to a mean
field thermal solution, the initial mode distribution for the dashed and the dashed-dotted
lines deviate more and more substantially from a thermal equilibrium distribution. It is
striking to observe that propagator modes with very different initial values but with the
same momentum p approach the same large-time value. The asymptotic behavior of the
two-point function modes are universal and uniquely determined by the initial energy
density.
One observes that after an effective damping of rapid oscillations the modes are
still far from equilibrium. All correlation functions quickly approach an exponentially
damped behavior. A characteristic rate γ0
 (damp)
 can be obtained from the zero mode
of the unequal-time two-point function F(t, 0; p = 0) with a corresponding time scale
proportional to the inverse rate. In this early-time range correlations with the initial
time are effectively suppressed and asymptotically F(t, 0; p = 0) → 0+. The early-
Introduction to Nonequilibrium Quantum Field Theory
 68
time range is followed by a smooth “drifting” of modes, which is characterized by
a slow dependence of F(t,t ′; p = 0) on (t + t ′)/2. The presence of such a regime is
a prerequisite for descriptions based on gradient expansions in the center coordinate
(t + t ′ )/2. Clearly, the early-time behavior for times ∼ 1/γ0
 (damp)
 is beyond the scope of
a gradient expansion. Though the exponential damping at early times is crucial for an
effective loss of details of the initial conditions, it does not determine the time scale for
thermalization. One typically finds very different rates for damping and for the late-time
approach of F(t,t; p = 0) to thermal equilibrium.
One of the “key” properties for the success of the quantum field theoretical description
is the nontrivial (i.e. not the free-field or “δ ”-type) dynamical spectral function ρ .
To analyze the spectral function (not to solve the dynamics!), we perform a Wigner
transformation for the modes ρ (t,t ′; p) and write with X 0 = (t + t ′ )/2, s0 = t − t ′:
iρ (X 0
 ; ω , p) =
 Z−2X 2X 0
 0
 ds0 eiω s0
 ρ (X 0 + s0/2, X 0 − s0 /2; p) .
 (215)
The i is introduced such that ρ (X 0 ; ω , p) is real. Because the spectral function is an-
tisymmetric, ρ (X 0; ω , p) = −ρ (X 0 ; − ω , p), and we will present the positive-frequency
part only. Since we consider an 0 initial-value problem with t,t ′ ≥ 0 , the time integral over
s0 = t − t ′ is bounded by ±2X .
On the right of Fig. 6 we display the Wigner transform for the zero momentum mode
as a function of ω . One clearly observes that the interacting theory has a continuous
spectrum described by a peaked spectral function with a nonzero width. The peak is
located at ω0 /mR = 1.46 in units of the initial renormalized mass (cf. Sec. 4.1.1), and
the results are shown for mRX 0 = 35.1 with λ /m2 R = 4.20 The inset shows a blow-up
around 3ω0 /mR = 4.38. The expected bump in the spectral function is small but visible.
We stress that this bump in the spectral function is kinematically forbidden for the “on-
shell” approximation and arises from “off-shell” 1 ↔ 3 processes. In Fig. 6 we also
s
’ Damping ’
 ’ Drifting’
 ’ Thermalization ’
 n o itidnocs
 n exponential suppression of smooth, parametrically slow
 approach to quantum
 lao ii t i correlations with initial time change of modes
 thermal equilibrium
 t i nd in fl c o rate:
 γ (damp)
 weak dependence of F(t,t’)
 rate: γ (therm)
 =
 γ (damp)
 o ssai ot i on t + t’
 ln I F(t,0)= φ(t) φ(0)
 0+
 F(t,t’)
 F (eq)
 (t−t’)
 e vitceffEearly
 intermediate
 late
 time
20
Here we used a “tsunami” similar to the one discussed in Sec. 4.1.1, however, the shown results are
insensitive to the initial condition details.
Introduction to Nonequilibrium Quantum Field Theory
 69
present fits to a Breit-Wigner spectral function
2ω Γp (X 0 )
ρBW(X 0; ω , p) =
 ,
 (216)
[ω 2 − ωp 2 (X 0 )] 2 + ω 2 Γ2 p (X 0 )
with a width Γp(X 0) = 2γp
 (damp)
 (X 0). While the position of the peak can be fitted easily,
the overall shape and width are only qualitatively captured. In particular, the slope of
ρ (X 0 ; ω , p) for small ω is quantitatively different. We also see that the Breit-Wigner fits
give a narrower spectral function (smaller width) and therefore would predict a slower
exponential relaxation in real time.
The characteristic time scales observed for the nonequilibrium evolution of modes in
Fig. 6 can be associated to
1. rapid oscillations of correlation functions with period ∼ 1/ωp described by the
“peak” of the spectral function.
2. damping of oscillations with inverse rate 1/γp
 (damp)
 described by a nonzero “width”
Γp = 2γp
 (damp)
 of the spectral function. In equilibrium the “width” is given by
2ω Γ(eq) ( ω , p) ≡ −Σρ (eq)
 (ω , p) ∼ O(λ 2/N) .
 (217)
3. late-time thermalization with inverse rate 1/γp
 (therm)
 because of “off-shell” number
changing processes. For the three-loop or NLO in 1/N approximation of the 2PI
effective action the processes changing the total particle number are perturbatively
of order ∼ λ 4 /N 2 (“slow!”). This can be understood from the fact that the total
particle number changing 2 processes require a nonzero “width” ∼ λ 2/N, and this
width enters in O( λ /N) evolution equations for the two-point functions.
One finds qualitatively the same characteristic ranges for the corresponding nonequilib-
rium evolution in 3 + 1 dimensions, and we will consider results for the chiral “quark-
meson” model below. An important quantitative difference between one and three spatial
dimensions results from the fact that “on-shell” 2 ↔ 2 processes contribute for the latter.
However, as for the 1 + 1 dimensional case they do not change the total “quasiparticle”
number. Total number changing “off-shell” or “on-shell” processes are required in gen-
eral to reach thermal equilibrium. As we have seen above, “off-shell” number changing
processes appear at NLO in the 2PI 1/N expansion. In contrast, they can be achieved
“on-shell” first at NNLO. E.g. for φ 4 -theory the lowest order “on-shell” (2 → 4) contri-
bution appears from the five-loop diagram (cf. Sec. 2.1):
Perturbatively, this contribution is of the same order ∼ λ 4 /N 2 than the “off-shell”
particle number changing processes arising at three-loop or NLO in the 1/N expansion
of the 2PI effective action. In this respect it is interesting to observe that precision tests
for the nonequilibrium dynamics as discussed in Sec. 5 indicate accurate results already
at NLO in the 2PI expansion.
Introduction to Nonequilibrium Quantum Field Theory
 70
-1
Time [m ]
0
 5
 10
 15
 20 25 30 50 100
0.55
e
d 0.5
omr e 0.45
preb 0.4
mun 0.35
no i Initial fermion distribution
 p=0.32 m A
t 1
a p 0.3
 A
 B
 B
u 0.5
 p=0.78 m A
cc B
0.25
O0
 p=1.38 m A
0
 p [m]
 2
 0
 p [m]
 2
 B
0.2
500
FIGURE 7. Fermion occupation number n( f ) (t; p) for three different momentum modes as a function of
time. The evolution is shown for two different initial conditions with same energy density. The long-time
behavior is shown on a logarithmic scale for t ≥ 30 m−1.
4.2. Prethermalization
Prethermalization is a universal far-from-equilibrium phenomenon which describes
the very rapid establishment of an almost constant ratio of pressure over energy density
(equation of state), as well as a kinetic temperature based on average kinetic energy. The
phenomenon occurs on time scales dramatically shorter than the thermal equilibration
time. As a consequence, prethermalized quantities approximately take on their final
thermal values already at a time when the occupation numbers of individual momentum
modes still show strong deviations from the late-time Bose-Einstein or Fermi-Dirac
distribution.
Here we consider the nonequilibrium evolution of quantum fields for a low-energy
quark-meson model, which is described in Sec. 2.3.1. It takes into account two quark
flavors with a Yukawa coupling ∼ h to a scalar σ -field and a triplet of pseudoscalar pions,
~ π . The theory corresponds to the well-known “linear σ -model”, which incorporates the
chiral symmetries of massless two-flavor QCD. The employed couplings in the action
(68) with (69) are taken to be of order one, and if not stated otherwise h = λ = 1. We
emphasize that the main results about prethermalization are independent of the detailed
values of the couplings. Here we employ the 2PI effective action (70) to two-loop order
given by (74). 21 All quantities will be given in units of the scalar thermal mass m.22
Thermalization: In Fig. 7 we show the effective occupation number density of fermion
21
 For the relation to a nonperturbative expansion of the 2PI effective action to next-to-leading order in
N f see Secs. 2.3.1 and 2.4).
22 The thermal mass m is evaluated in equilibrium. It is found to prethermalize very rapidly. The employed
spatial momentum cutoff is Λ/m = 2.86.
Introduction to Nonequilibrium Quantum Field Theory
 71
16
]
m[erutarepmetedoM8
4
2
1
t = 100 : fermion
t = 200 : fermion
t = 400 : fermion
scalar
scalar
scalar
late time
0.5
0
 0.5
 1
 1.5
 2
 2.5
Momentum [m]
( f ,s)
FIGURE 8. Fermion and scalar mode temperatures Tp
 (t) as a function of momentum p for various
times.
momentum modes, n( f ) (t; p), as a function of time for three different momenta 23. The
plot shows two runs denoted as (A) and (B) with different initial conditions but same
energy density. Run (A) exhibits a high initial particle number density in a narrow
momentum range around ±p. This situation is reminiscent of two colliding wave packets
with opposite and equal momentum. We emphasize, however, that we are considering
a spatially homogeneous and isotropic ensemble with a vanishing net charge density.
For run (B) an initial particle number density is employed which is closer to a thermal
distribution.
One observes that for a given momentum the mode numbers of run (A) and (B)
approach each other at early times. The characteristic time scale for this approach is well
described by the damping time tdamp(p) 24 . Irrespective of the initial distributions (A) or
(B), we find (for p/m ≃ 1) tdamp ( f )
 ≃ 25 m−1 for fermions and tdamp (s)
 ≃ 28 m−1 for scalars.
In contrast to the initial rapid changes, one observes a rather slow or “quasistationary”
subsequent evolution. The equilibration time teq ≃ 95 m−1 is substantially larger than
tdamp and is approximately the same for fermions and scalars. Thermal equilibration is
a collective phenomenon which is, in particular, rather independent of the momentum.
As we have also observed for the 1 + 1 dimensional scalar theory in Sec. 4.1.4, mode
quantities such as effective particle number distribution functions show a characteristic
two-stage loss of initial conditions: after the damping time scale much of the details
about the initial conditions are effectively lost. However, the system is still far from
equilibrium and thermalization happens on a much larger time scale.
23
 This quantity is directly related to the expectation value of the vector component of the field commu-
24 tator h[ψ , ψ̄ ]i in Wigner coordinates and fulfills 0 ≤ n(f) (t; p) ≤ 1.
The rate 1/t
damp (p) is determined by the spectral component of the self-energy.
Introduction to Nonequilibrium Quantum Field Theory
 72
ε
/p=wetatsfonoitauqE0.6
0.4
0.2
0.5
ε
/p=w0.3
0.1
0
h = 1.0
h = 0.5
-12
 Time [m ]
 8
0
0
 5
 10
 15
 20
 25
 30 50
 100
-1
Time [m ]
500
FIGURE 9. The ratio of pressure over energy density w as a function of time. The inset shows the early
stages for two different couplings and demonstrates that the prethermalization time is independent of the
interaction details.
( f ,s)
We define mode temperatures Tp (t) by equating the mode particle numbers
( f ,s)
n p (t) with a time and momentum dependent Bose-Einstein or Fermi-Dirac distribu-
tion, respectively:
n p(t) = !
 [exp (ω p (t)/T p(t)) ± 1]−1 .
 (218)
This definition is a quantum mechanical version of its classical counterpart
 as defined by
T the p = squared T eq equation “generalized (218) yields velocities”. the familiar In thermal occupation equilibrium numbers with (μω = p 0). ≃ p Here p2 + the M2 mode
 and
frequency ω ( p f ,s)
(t) is determined by the peak of the spectral function for given time and
momentum, as detailed for the scalar theory in Sec. 4.1.4. In Fig. 8 we show the fermion
and scalar mode temperature as a function of momentum for various times t ≫ tdamp .
One observes that at late times, when thermal equilibrium is approached, all fermion
( f )
 (s)
and scalar mode temperatures become constant and agree: Tp (t) = Tp (t) = T eq . In
contrast, there are sizeable deviations from the thermal result even for times considerably
larger than the characteristic damping time.
Kinetic prethermalization: In contrast to the rather long thermalization time, prether-
malization sets in extremely rapidly. In Fig. 9 we show the ratio of pressure over energy
density, w = p/ε , as a function of time. One observes that an almost time-independent
equation of state builds up very early, even though the system is still far from equilib-
rium! The prethermalization time tpt is here of the order of the characteristic inverse
mass scale m−1. This is a typical consequence of the loss of phase information by sum-
ming over oscillating functions with a sufficiently dense frequency spectrum. In order
to see that this phenomenon is not related to scattering or to the strength of the interac-
tion, we compare with a smaller coupling in the inset and observe good agreement of
Introduction to Nonequilibrium Quantum Field Theory
 73
1.8
s
erutarepmetevitceffE1.6
1.4
1.2
1
0.8
0.6
Tkin/Teq
Tchem/Teq: h=1.0, Teq/m=1.0
h=1.0, Teq/m=2.8
h=0.7, Teq/m=2.1
0.4
0
 100
 200
 300
 400
 500
-1
Time [m ]
FIGURE 10. Chemical temperatures for scalars (upper curves) and fermions (lower curves) for different
values of the coupling h and T eq. We also show the kinetic temperature T kin (t) (solid line), which
prethermalizes on a very short time scale as compared to chemical equilibration.
both curves. The dephasing phenomenon is unrelated to the scattering-driven process of
thermalization.
Given an equation of state, the question arises whether there exists a suitable definition
of a global kinetic temperature T kin . In contrast to a mode quantity such as Tp (t), a
temperature measure which averages over all momentum modes may prethermalize.
Building on the classical association of temperature with the mean kinetic energy per
degree of freedom, we use here a definition based on the total kinetic energy Ekin(t):
T kin (t) = Ekin (t)/ceq .
 (219)
Here the extensive dimensionless proportionality constant ceq = Ekin,eq/T eq is given
solely in terms of equilibrium quantities25. Since total energy is conserved, the time scale
when “equipartition” is reached (i.e. Ekin /E is approximately constant) also corresponds
to a time-independent kinetic temperature. The latter equals the equilibrium temperature
T eq if Ekin/E has reached the thermal value.
The solid line of Fig. 10 shows T kin (t) normalized to the equilibrium temperature
(for T eq /m = 1). One observes that an almost time-independent kinetic temperature is
established after the short-time scale tpt ∼ m−1. The time evolution of bulk quantities
such as the ratio of pressure over energy density w, or the kinetic temperature T kin , are
dominated by a single short-time scale. These quantities approximately converge to the
thermal equilibrium values already at early times and can be used for an efficient “quasi-
25
 For a relativistic plasma one has Ekin /N = ε /n = α T . As alternatives, one may consider the weighted
average T̄ (t) = ∑ n(t; p)T (t; p)/ ∑ n(t; p) where the sum is over all modes, or a definition analogous to
Eq. (220) below.
Introduction to Nonequilibrium Quantum Field Theory
 74
thermal” description in a far-from-equilibrium situation!
Chemical equilibration: In thermal equilibrium the relative particle numbers of dif-
ferent species are fixed in terms of temperature and particle masses. A system has chem-
ically equilibrated if these ratios are reached, as observed for the hadron yields in heavy
ion collisions. Obviously, the chemical equilibration time t ch will depend on details of
the particle number changing interactions in a given model and tch ≤ teq . In our model
we can study the ratio between the numbers of fermions and scalars. For this purpose
( f ,s)
we introduce the chemical temperatures T (t) by equating the integrated number
chEinstein/Fermi-Dirac density of each species, form n( f ,s) of (t) distributions:
 = g( f ,s) R
 d3 p/(2π )3 n( p
 f ,s)
(t), with the integrated Bose-
n(t) = !
 2π
 g
 2
 Z0
 ∞
 dpp2 [exp (ω p (t)/T ch(t)) ± 1]−1 .
 (220)
Here g( f ) = 8 counts the number of fermions and g(s) = 4 for the scalars.
(s, f )
The time evolution of the ratios T ch (t)/T eq is shown in Fig. 10 for different values of
the coupling constant h and the equilibrium temperature T eq . One observes that chemical
(s)
 ( f )
equilibration with T (t) = T (t) does not happen on the prethermalization time scale,
ch chin contrast to the behavior of T kin(t). Being bulk quantities, the scalar and fermion
chemical temperatures can approach each other rather quickly at first. Subsequently, a
slow evolution towards equilibrium sets in. For the late-time chemical equilibration we
find for our model tch ≃ teq. However, the deviation from the thermal result can become
relatively small already for times t ≪ teq.
Let us finally consider our findings in view of collisions of heavy nuclei and try to
estimate the prethermalization time. Actually, tpt is rather independent of the details of
the model like particle content, values of couplings etc. It mainly reflects a characteristic
frequency of the initial oscillations. If the “temperature” (i.e. average kinetic energy
per mode) sets the relevant scale one expects T tpt = const. (For low T the scale will
be replaced by the mass.) For our model we indeed find T tpt ≃ 2 − 2.5. 26 We expect
such a relation with a similar constant to hold for the quark-gluon state very soon after
the collision. For T & 400 − 500 MeV we obtain a very short prethermalization time tpt
of somewhat less than 1 fm. This is consistent with observed very early hydrodynamic
behavior in collision experiments, however, one further has to investigate the required
isotropization of pressure.
4.3. Far-from-equilibrium field dynamics: Parametric resonance
In classical mechanics parametric resonance is the phenomenon of resonant amplifi-
cation of the amplitude of an oscillator having a time-dependent periodic frequency. In
the context of quantum field theory a similar phenomenon describes the amplification
of quantum fluctuations, which can be interpreted as particle production. It provides an
26
 We define tpt by |w(tpt ) − weq |/weq < 0.2 for t > t pt.
Introduction to Nonequilibrium Quantum Field Theory
75
important building block for our understanding of the (pre)heating of the early universe
at the end of an inflationary period, and may also be operative in various situations in
the context of relativistic heavy-ion collision experiments. Here we will consider the
phenomenon as a “paradigm” for far-from-equilibrium dynamics of macroscopic fields
or one-point functions. Dynamics of correlation functions for vanishing — or similarly
for constant — macroscopic fields has been described in detail above. There is a wealth
of phenomena associated to non-constant fields such as parametric resonance, spinodal
decomposition or in general the dynamics of phase transitions where the field can play
the role of an order parameter.
The example of parametric resonance is particularly challenging since it is a nonper-
turbative phenomenon even in the presence of arbitrarily small couplings. Despite being
a basic phenomenon that can occur in a large variety of quantum field theories, paramet-
ric resonance is a rather complex process, which in the past defied most attempts for a
complete analytic treatment even for simple theories. It is a far-from-equilibrium phe-
nomenon that involves densities inversely proportional to the coupling. The nonpertur-
batively large occupation numbers cannot be described by standard kinetic descriptions.
So far, classical statistical field theory simulations on the lattice have been the only quan-
titative approach available. These are expected to be valid for not too late times, before
the approach to quantum thermal equilibrium sets in (cf. also Sec. 5). Studies in quan-
tum field theory have been mainly limited to linear or mean-field type approximations
(leading-order in large-N, Hartree), which present a valid description for sufficiently
early times. However, they are known to fail to describe thermalization and miss impor-
tant rescattering effects (cf. also Sec. 4.1). The 2PI 1/N expansion provides for the first
time a quantitative nonperturbative approach in quantum field theory taking into account
rescattering.
Recall for a moment the classical mechanics example of resonant amplitude amplifi-
cation for an oscillator with time-dependent periodic frequency. The amplitude y(t) is
described by the differential equation ÿ + ω 2 (t)y = 0 with periodic ω (t + T ) = ω (t) of
period T . Since the equation is invariant under t → t + T there are periodic solutions
y(t + T ) = c y(t). This can be expressed as y(t) = ct/T Π(t) with periodic Π(t + T ) =
Π(t). One concludes that for real c > 1 there is an instability with an exponential growth.
For small elongations a physical realization of this situation is a pendulum with a peri-
odically changing length as displayed:
periodic
In contrast to the mechanics example, in quantum field theory there will be no external
periodic source. A large coherent field amplitude coupled to its own quantum fluctua-
tions will trigger the phenomenon of parametric resonance. Mathematically, however,
important aspects are very similar to the above classical example for sufficiently early
times: The mechanical oscillator amplitude y plays the role of the statistical two-point
function F in quantum field theory, and the periodic ω 2 (t) plays the role of an effec-
Introduction to Nonequilibrium Quantum Field Theory
 76
tive mass term M 2 (φ (t)) whose time dependence is induced by an oscillating macro-
scopic field φ (t). Simple linear approximations to the problem, which have been much
employed in the literature, are even mathematically equivalent to the above mechanics
example. Accordingly, the well-known Lamé–type solutions of the mechanics problem
will also play a role in the quantum field theory study. Substantial deviations do, how-
ever, quickly set in with important non-linear effects.
4.3.1. Parametric resonance in the O(N) model
We consider the scalar N-component quantum field theory with classical action (13),
and employ the 2PI 1/N-expansion to next-to-leading order. The relevant equations of
motion are given by (183)–(189) as described in Sec. 3.6.1. We will describe numerical
solutions of these equations in 3 + 1 dimensions without further approximations. More-
over, the approach allows us to identify the relevant contributions to the dynamics at
various times and to obtain an approximate analytic solution of the nonlinear dynamics
for the entire amplification range. It should be emphasized that the approach solves the
problem of an analytic description of the dynamics at nonperturbatively large densities.
It is necessary to take into account the infinite set of NLO 2PI diagrams as described
in Sec. 2.4.1. As we will show in the following, each of these eventually contributes to
the same order in the coupling λ such that any finite order in loops or couplings is not
sufficient. In this sense, the 2PI 1/N-expansion to NLO represents a minimal approach
for the controlled description of the phenomenon. This justifies the rather involved com-
plexity of the approximation.
We have in mind a situation reminiscent of that in the early universe after a period of
chaotic inflation, driven by a macroscopic (inflaton) field. We consider a weakly coupled
system that is initially in a pure quantum state, characterized by a large field amplitude
φa (t) = σ (t)M 0
r
 6N
 λ
 δa1 .
 (221)
and small quantum fluctuations, corresponding to vanishing particle numbers at initial
time. Here M0 sets our unit of mass, σ (0) = σ0 is ∼ O(1) and ∂t σ (t)| t=0 = 0. The initial
statistical propagator contains a “longitudinal” component F k and (N − 1) “transverse”
components F :
⊥F ab = diag{F k , F ⊥ , . . ., F ⊥} ,
 (222)
with ∂t F(t, 0; p)|t=0 = 0, and ∂t ∂t ′ F(t,t ′; p)|t=t ′ =0 ≡ F −1 (0, 0; p)/4 for a pure-state ini-
tial density matrix (cf. Sec. 3.2).
To get an overview, in Fig. 11 we show the contributions from the macroscopic field
and the from the fluctuations to the (conserved) total energy as functions of time for
N = 4 and λ = 10−6 . 27 The total energy Etot is initially dominated by the classical part
27
 Typical volumes (Ns as )3 of Ns = 36–48 with as = 0.4–0.3 lead to results that are rather insensitive to
finite-size and cutoff effects.
Introduction to Nonequilibrium Quantum Field Theory
 77
1.0
)
0(E/)t(E0.5
tnonpert
E fluc
E cl
0.0
0
 20
 40
 60
 80
 100
M0t
FIGURE 11. Total energy (solid line) and classical-field energy Ecl (dotted line) as a function of time.
The dashed line represents the fluctuation part Efluc , showing a transition from a classical-field to a
fluctuation dominated regime.
Ecl (t = 0), with reads in terms of the rescaled field (221):
Ecl L3
 (t)
 =
 3N λ M 0
 2
 
∂ t ∂ t ′ + m2
 + 2
 1 M 0 2σ 2
 (t)
 σ (t)σ (t ′ )
 t=t ′
 ,
 (223)
where L3 denotes the spatial volume, and Efluc = Etot − E cl. More and more energy
is converted into fluctuations as the system evolves. A characteristic time — denoted
as tnonpert in Fig. 11 — is reached when both contributions become of the same size,
i.e. Efluc ≃ Ecl . Before this time, the coherent oscillations of the field φ lead to a resonant
enhancement of the statistical propagator Fourier modes
F (t,t ′; p0 ) eγ0(t+t ′
 ) .
 (224)
⊥ ∼in a narrow range of momenta around a specific value |p| ≃ p0 : this is parametric reso-
nance. Apart from the resonant amplification in the linear regime, we identify two char-
acteristic time scales — denoted as tsource and tcollect below —, which signal strongly en-
hanced particle production in a broad momentum range. Nonlinear interactions between
field modes cause the resonant amplification to spread to a broad range of momenta.
More specifically, the initially amplified modes act as a source for other modes. The rate
of the source-induced exponential amplification exceeds the characteristic rate γ0 for the
resonant amplification. This is illustrated in Figs. 12 and 13, where the effective parti-
cle numbers are shown for various momenta as a function of time in the transverse and
the longitudinal sector. Before the transition to a fluctuation dominated regime around
tnonpert , one observes a rapid change of the particle numbers due to resonant as well as
source-induced amplification. The source-induced amplification is crucial for the rapid
approach to the subsequent, quasistationary regime, where direct scattering drive a very
slow evolution towards thermal equilibrium. The transition to this slow regime around
tnonpert can be very well observed from Figs. 12 and 13.
The fluctuation dominated regime is characterized by strong nonlinearities. For in-
stance, from Fig. 11 one infers for t ≃ t nonpert that the classical field decay “overshoots”
Introduction to Nonequilibrium Quantum Field Theory
 78
8
10
)
p,t(T
n5
10
2
10
2
tcollect
0
6
0
+
0
p0
2p0
3p 0
4p0
5p0
−1
10
0
 20
 40
 60
 80
M t
0FIGURE 12. Effective particle number density for the transverse modes as a function of time for various
momenta p ≤ 5p0. At early times, modes with p ≃ p0 are exponentially amplified with a rate 2γ0 . Due to
nonlinearities, one observes subsequently an enhanced growth with rate 6γ0 for a broad momentum range.
8
10
)
p,t(||n5
10
2
10
tsource
4
0
+
0
p0
2p0
3p0
4p0
5p0
−1
10
0
 20
 40
 60
 80
M 0 t
FIGURE 13. Same as in Fig. 12, for the longitudinal modes. Nonlinear source effects trigger an
exponential growth with rate 4γ0 for p . 2p0 . The thick line corresponds to a mode in the parametric
resonance band, and the long-dashed line for a similar one outside the band. The resonant amplification is
quickly dominated by source-induced particle production.
and is temporarily reversed by feed-back from the modes. This can be directly seen in
the evolution for the rescaled field shown in Fig. 14, which shows a dip around tnonpert .
(The particle numbers of Figs. 12 and 13 exhibit correspondingly a reverse behavior.)
The oscillations in the envelope of σ (t) damp out with time, and one observes a slow
decay of the field and the associated energy at later times. The latter phenomena cannot
be seen in leading-order or Hartree–type approximations and it is crucial to include the
next-to-leading order contributions (cf. also Sec. 4.1.1). They are important for a reliable
description of the system at the end of the resonance stage for finite N . 1/λ . For real-
istic inflationary models with typically λ ≪ 1 this is, in particular, crucial to determine
whether there are any radiatively restored symmetries.
Introduction to Nonequilibrium Quantum Field Theory
 79
)
t(σ2.0
0.0
−2.0
0
 20
 40 tnonpert 60
 80
 100
M0t
FIGURE 14. The rescaled field σ as a function of time.
The characteristic properties described above can be understood analytically from the
evolution equations for the one- and two-point functions. To set the scale we use the
initial longitudinal mass squared M 0 2 ≡ M2(t = 0) with (cf. Eq. (146)):
M2(t) = m2 Λ +
d 6N
 3
 λ p
 h
3T k(t) + (N − 1)T ⊥(t)i
 ,
 (225)
T k,⊥(t) =
 Z (2π )3 F k,⊥
 (t,t; p) ,
 (226)
where we denote the “tadpole” contributions by T and T from the longitudinal
k ⊥and transverse propagator
 components, respectively, 2 and 2 some Λ 2≫ 1/2
 p0 . Initially,
larly F k (0, for 0; p) the = transverse 1/ 2 2ωk (p) components  with frequency F ⊥ (0, ω 0; k (p) p) where = [p + the M0 frequency (1 + 3σ0 ω )] ⊥(p) , and contains
 simi-
a mass term σ0 instead of 3σ0 2. Parametrically the initial statistical propagator is of
order one, i.e. ∼ O(N 0λ 0 ). Parametric resonance leads to the dominant amplification
of F (t,t; p0) with rate 2γ0 . As a consequence of the exponential amplification of
⊥the statistical propagator there is a characteristic time at which loop corrections will
become of order one as well. This is schematically summarized in Fig. 15. The time
when F ⊥ (t,t; p0 ) ∼ O(N 0 λ −1/2 ) is denoted by t = tsource. At this time the one-loop
diagram with two field insertions indicated by crosses as depicted in Fig. 15 will give a
contribution of order one to the evolution equation for F (t,t; p). For instance, the two
kpowers of the coupling coming from the vertices of that diagram are canceled by the
field amplitudes (221) and by propagator lines associated to the amplified F (t,t ′; p0 ).
⊥Similarly, at the time t = tcollect the maximally amplified transverse propagator mode has
grown to F ⊥(t,t ′ ; p0 ) ∼ O(N1/3λ −2/3 ). As a consequence, the “setting sun” diagram
in Fig. 15 becomes of order one and is therefore of the same order as the classical
contributions. Though the loop corrections become of order one later than the initial
time, they induce amplification rates that are multiples of the rate γ0 which lead to a
very rapid growth of modes in a wide momentum range. Finally, when the fluctuations
have grown nonperturbatively large with F ⊥(t,t ′; p0 ) ∼ O(N 0 λ −1) any loop correction
will no longer be suppressed by powers of the small coupling λ . In this case the nonper-
turbative 1/N expansion becomes of crucial importance for a quantitative description of
the dynamics to later times. In the following, we derive these results from the evolution
equations (183)–(189) in more detail.
Introduction to Nonequilibrium Quantum Field Theory
 80
time
:
emigerevitabrutrepnonn
oituloveyranoitatsisauq(IV)
~N, N 0;
+
tnonpert ~ ln ( −1 )/2
F ~ O(N 0 −1 )
0
+
~N 0
+
+
~ O(
 −1 )
~ O(N 0
 0 )
:
emigerraenilnonn
oitacifilpmadecudniecruos:
emigerraenilecnanosercirtemarap(III)
tcollect ~ 2 t nonpert /3 + ln(N)/ 60
1/3 −2/3 −1
F ~ O(N ) for N <
~
rate:
 6
~ O(N 0
 0 )
0
 for F (p = / p 0 )
(II)
tsource ~ tnonpert /2
0
 −1/2
F ~ O(N )
(I)
F (t,t;p0 ) ~ exp( 2
0
t)
x
 x
~ O(N 0
 0 )
rate:
 4
 for F (p <
 2p )
0
 ~ 0rate:
 2
0
t = 0 , F ~ O(N 0
 0 )
FIGURE 15. Schematic overview of the characteristic time scales and the respective relevant diagram-
matic contributions (see text).
(I) Early-time (linear) regime: Resonant amplification. At early times the σ –field
evolution equation receives the dominant, i.e. O(λ 0 ), contributions from the classical
action S given in (13). As a consequence, the field dynamics can be described by the
classical equation of motion. In the following, all quantities are rescaled with appropriate
powers of M0 to become dimensionless and we set M0 ≡ 1. The classical field equation
reads
∂t 2σ (t) + σ (t) + σ 3(t) = 0 .
 (227)
Differential equations of this time have been extensively studied in the litera-
ture. For the initial condition considered here it has the (anti-)periodic solution
σ (t + π /ω0 ) = −σ (t), which can be expressed in term of the Jacobian cosine cn
oscillating as σ (t) = behavior σ0 cn[t q
 with 1 + σ0 characteristic 2 , σ0/q
2(1 + frequency
 σ0 2 )]. Therefore, the solution shows a rapidly
ω0 ≃ 2q
1 + σ0 2 .
 (228)
Introduction to Nonequilibrium Quantum Field Theory
 81
The period average of the field amplitude can be also expressed in terms of the initial
amplitude σ0 as σ 2 (t) ≃ σ0 2 /2.
 0The evolution equations for the two-point functions to order O(λ ) correspond to
free-field equations with 2 the addition of a time-dependent mass term ∼ 3σ 2 (t) for the
longitudinal and ∼ σ (t) for the transverse modes:
h
∂t 2 2
 + p2
 + 1 + σ 2
 (t)i
F ⊥ (t,t ′; p) = 0
h
∂t + p2 + 1 + 3σ 2(t)i
F k (t,t ′; p) = 0 ,
 (229)
and equivalently for the spectral functions ρ (t,t ′; p) and ρ (t,t ′ ; p). As a consequence
⊥ kof this approximation, the two-point functions can be factorized as products of single-
time functions:
ρ F ⊥(t,t ′ ′ ; ; p) p) = =
 i 2 1 
 f f ⊥(t; p) p) f f ∗ ⊥ ∗ (t (t ′ ; ′p) ; p) + f ∗ f⊥ ∗ (t; (t; p) p) f f ⊥(t (t ′; ′p) ; p)
 ,
 ,
 (230)
and similarly for ⊥(t,t the longitudinal 
 components. ⊥(t; ⊥ We − emphasize ⊥ ⊥ that the 
 simple decom-
position (230) is no longer correct at higher orders in the coupling. In terms of these
so-called mode functions the equations of motion read, e.g. for the transverse modes:
h
∂t 2
 + p 2
 + 1 + σ 2
(t)i
 f⊥ (t; p) = 0 .
 (231)
Up to an overall arbitrary phase,
 the above initial conditions
 for the two-point functions
for translate fk. For into this f ⊥ approximation (0, p) = 1/p2ω the ⊥ (p), quantum ∂ t f⊥ (t, field p)|t=0 theory = −i problem pω⊥ (p)/2 becomes and equivalently
 mathemati-
cally equivalent to the well-known classical mechanics problem described in the above
introduction. The analytical solution of the Lamé–type equation (231) is well known and
can be summarized for our purposes as follows: There is an exponential amplification
of F ⊥ (t,t ′; p) for a bounded momentum range 0 ≤ p2 ≤ σ0 2 /2. This corresponds to a
maximum momentum for amplification p2max + 1 + σ 2 (t) ≃ 1 + σ0 2 = (ω0 /2)2. A further
important result is that there is a separation 2 of scales: ω0 ≫ γ0 , with the maximum am-
plification rate γ0 ≃ 2δ ω0 for p2 = p20 ≃ σ0 /4 with the small number δ ≤ e−π ≃ 0.043.
There is much smaller growth in a narrow momentum range for F k, which is of no
importance here since it is quickly dominated by loop corrections as is shown below.
Time-averaged over ∼ ω0 −1 , for t,t ′ ≫ γ0 −1 one finds the result of Eq. (224). Note here
that also leading-order large-N or Hartree–type approximations show the same bounded
amplification regime, such that at all times no higher momentum modes would get pop-
ulated. This is, of course, an artefact of the approximation which is absent once NLO
corrections are taken into account as is described below.
The analytic results for the Lamé regime agree precisely with the NLO numerical
results for sufficiently early times. Figs. 12 and 13 show the transverse and longitudinal
particle numbers for various momenta in the range 0 < p ≤ 5p0, averaged over the rapid
oscillation time ∼ 1/ω0 . For the transverse modes the lowest two momenta shown are
Introduction to Nonequilibrium Quantum Field Theory
 82
inside the resonance band. We define the effective particle numbers as in Eq. (201).28 As
shown in Sec. 4.1.2 for the O(N) model these definitions yield an efficient description
approaching a Bose-Einstein distributed particle number at sufficiently late times.
(II) Source-induced (nonlinear) amplification regime: Strongly enhanced particle pro-
duction for longitudinal modes. The O(λ 0 ) approximation for longitudinal modes
breaks down at the time
t ≃ t ′ = tsource : F ⊥ (t,t ′ ; p0 ) ∼ O(N 0λ −1/2) .
 (232)
This can be derived from the O(λ ) evolution equations to which one-loop self-energies
x
 x
contribute, which diagrammatically are given by
 ,
 . The approximate evolu-
tion equation reads:

∂t 2 + p2 + M 2 (t) + 3σ 2(t) t
 
F k (t,t ′ ; p) ≃
2λ (N 3N
 − 1)
 σ (t)
( Z
 0
 dt ′′ σ (t ′′ )Π ρ
 ⊥(t,t ′′; p)F k (t ′′,t ′ ; p)
−
 1 2 Z0
t ′
 dt ′′σ (t ′′)ΠF
 ⊥ (t,t ′′ ; p)ρk (t ′′,t ′ ; p))
 ≡ RHS .
 (233)
Here A = {F, we ρ }, have and abbreviated we used that ΠA ⊥ Π (t,t ⊥ ≫ ′′; p) Πk = and R
 d3q/(2π F ⊥ 2 ≫ ρ )3 ⊥
 2 .29 F ⊥(t,t One ′′; observes p − q) A⊥(t,t that indeed ′′; q) with
 for
F ⊥ ∼ O(N 0λ −1/2 ) the r.h.s. of Eq. (233) becomes ∼ O(1) and cannot be neglected.
In order to make analytical progress, one has to evaluate the “memory integrals” in
the above equation. This is dramatically simplified by the fact that the integral is ap-
proximately local in time, since the exponential growth lets the latest-time contributions
dominate the integral. (This will be the case for times t . tnonpert after which exponential
amplification stops, cf. below.) For the approximate evaluation of the memory integrals
we consider time-averages over ω0 −1 ≪ γ0 −1 :
Z0
 t
 dt
 ′′
 −→
 Zt−c/ω0
 t
 dt ′′
 (c ∼ 1)
 (234)
and perform a Taylor expansion around the latest time t (t ′):
ρk,⊥ (t,t ′′; p) ≃ ∂ t ′′ ρk,⊥(t,t ′′ ; p)|t=t ′′ (t ′′ − t) ≡ (t − t ′′ ) ,
F k,⊥ (t,t ′′; p) ≃ F k,⊥(t,t; p) ,
 (235)
28
 29
 We have employed Q ≡ 0 as in Sec. 4.1.2 which is, however, irrelevant.
The latter inequality will be discussed in detail in Sec. 5, where it is shown to indicate the validity of
classical statistical field theory approximations.
Introduction to Nonequilibrium Quantum Field Theory
 83
where we have used the equal-time commutation relations (117). With these approxima-
tions the r.h.s. of Eq. (233) can be evaluated as:
RHS λ σ 2 (t)
 c2 (N − 1)
 T (t)F (t,t ′; p)
 (mass term)
 (236)
≃ ω0
 2
 3N
 ⊥ k+ λ σ (t)σ (t ′)
 c2 (N − 1) ΠF
 (t,t ′; p)
 (source term)
 (237)
ω0 2 6N
 ⊥The first term is a contribution from NLO to the mass, whereas the second term repre-
sents a source. Note that both the LO mass term and the above correction to this mass
are of the same order in λ , however, with opposite sign. To evaluate the momentum
integrals, we use a saddle point approximation around the dominant p ≃ p0 , valid for
t,t ′ ≫ γ0 −1 , with F ⊥ (t,t ′, p) ≃ F ⊥(t,t ′ , p0 ) exp[−| γ0 ′′ |(t +t ′)(p − p0 )2 /2].30 From this one
obtains for the above mass term:
p20 F (t,t; p 0)
λ T ⊥(t) ≃ λ
 2(π ⊥ .
 (238)
3|γ0 ′′|t)1/2
The result can be used to obtain an estimate at what time t this loop correction becomes
an important contribution to the evolution equation. Note that to this order in λ it is cor-
rect O(1) to use can F then ⊥(t,tbe ′ ; p0 written ) ∼ eγ0 for (t+t ′
 ) on the 1 as:
 r.h.s. of (238). The condition λ T ⊥(t = tnonpert )
∼ λ ≪1
tnonpert ≃
 2γ0
 ln λ −1
 (239)
The same saddle point approximation can be performed to evaluate the source term
(237):
F
 p20 F 2 (t,t ′; p0)
λ Π⊥(t,t ′
 ; 0) ≃ λ
 4(π 3 ⊥ ′′ |(t + t ′))1/2
 .
 (240)
|γ0Here we only wrote the source term for p = 0 where it has its maximum, although it
affects all modes with p . 2p0 . Again this can be used to estimate the time t = tsource at
which λ Π F ⊥ ∼ O(1):
1
tsource ≃ 2
 t nonpert
 (241)
One arrives at the important conclusion that the source term (237) becomes earlier of
order one than the mass term (236): For tsource . t . tnonpert the source term dominates
the dynamics! Using these estimates in (233) one finds that the longitudinal modes with
0 . p . 2p0 get amplified with twice the rate 2γ0 :
F k (t,t; p) ∼ λ F ⊥ 2 (t,t; p0 ) ∼ λ e4γ0t .
 (242)
30
 Here γ (p) ≃ γ0 + γ0 ′′(p − p 0)2 /2 with γ0 ′′ ≃ −64δ (1 − 6 δ )q
 1 + σ0 2 /σ0 2 .
Introduction to Nonequilibrium Quantum Field Theory
84
Though the non-linear contributions start later, they grow twice as fast! The analytical
estimates for tsource and rates agree very well with the numerical results shown in Fig. 13.
(III) Collective amplification regime: explosive particle production in a broad mo-
mentum range for transverse modes. A similar analysis can be made for the transverse
modes. Beyond the Lamé–type O(λ 0 ) description, the evolution equation for F re-
⊥ceives contributions from the feed-back of the longitudinal modes at O(λ ) as well as
from the amplified transverse modes at O(λ 2 ). They represent source terms in the evo-
lution equation for F ⊥(t,t ′ ; p) which are both parametrically of the form ∼ λ 2 F ⊥ 3 /N as
is depicted below:
x
 x
 λ
 cf. (242) λ
 2
 3∼ N
 λ 2 F kF 3
 ⊥ ∼
 N F ⊥
 )
 ∼
 λ N
 2 e
6γ0 t
∼
 N F
 ⊥
Following along the lines of the above paragraph and using (239) this leads to the
characteristic time t = t collect at which these source terms become of order one:
2
 ln N
tcollect ≃
 3
 tnonpert +
 6γ0
(243)
For t ≃ t ′ ≃ tcollect the dominant transverse mode has grown to
F ⊥ (t,t ′; p0 ) ∼ O(N 1/3 λ −2/3 ) .
 (244)
Correspondingly, for tcollect . t . t nonpert one finds a large particle production rate ∼ 6γ0
in a momentum range 0 . p . 3p0 , in agreement with the full NLO results shown in
Fig. 12. In this time range the longitudinal modes exhibit an enhanced amplification
as well (cf. Fig. 13). It is important to realize that the phenomenon of source-induced
amplification repeats itself: the newly amplified modes, together with the primarily
amplified ones, act as a source for other modes, and so on. In this way, even higher
growth rates of multiples of γ0 can be observed and the “explosive” amplification
rapidly propagates towards higher momentum modes. We emphasize that the collective
amplification regime is absent in the LO large-N approximation. Consequently, even for
the transverse sector the latter does not give an accurate description if tcollect ≤ tnonpert ,
that is for N . λ −1.
Behavior of the field. Around t . tnonpert is the earliest time when sizeable correc-
tions to the classical field equation (227) appear, which give (cf. Eq. 189):
∂t 2
 + 1 + δ M2 (t) + σ 2 (t)
 σ (t) = −Z0
t
 dt ′Σk ρ
 (t,t ′ ; p = 0)|σ =0 σ (t ′) ,
 (245)
where δ M2 λ (N−1)
 6N T . Before tnonpert , where the “memory expansion” discussed above
≃ ⊥is valid, we can discuss the behavior of the field σ = σ (0) + δ σ perturbatively in terms of
Introduction to Nonequilibrium Quantum Field Theory
 85
a slowly varying small correction δ σ to the classical solution σ (0) of (227) and a small
δ M2. If we neglect for a moment the NLO contributions on the r.h.s. the linearized
equation (245) then yields
σ (t) ≃ 
 1 −
 1 + 1
 3σ0
 2
 λ (N 6N
 − 1)
 T ⊥ (t)
 σ (0) (t) .
 (246)
One concludes that there is an exponential decrease of the field amplitude at LO since
T ⊥ (t) ∼ e2γ0t for t . tnonpert according to (238). The dominant NLO corrections read
Zt−c/ω0
 t
 dt ′ Σk ρ
 (t,t ′ ; p = 0)| σ =0 σ (t ′ ) ≃
 2ω0 c2 2
 ∂
t Σ
k
 ρ
 (t,t
 ′
;
 p
 =
 0)|
σ =0
|t=t ′ σ (t)
O(λ ≃ 2 )
 − 2ω0 c22
 18N
 λ 2
 Z Λ (2π d3
 q
 ) 3 (2π d3k
 ) 3
 F ⊥(t,t; −q − k)F ⊥ (t,t; k) σ (t)
c2 λ 2 2
≃−
 2ω0 2 18N T ⊥
 (t) σ (t) .
 (247)
Note that the NLO correction to the effective mass term comes with an opposite sign
than the LO correction. For t tnonpert one has δ M2 O(N 0λ 0 ) and Σρ
 O(N−1λ 0 ),
→ ∼ k ∼i.e. they become of the same order in λ . As a consequence, cancellations may lead
(temporarily) to reverse field decay. This is indeed what one observes from the full
numerical NLO solution for N = 4 and λ = 10−6 shown in Fig. 14. Around tnonpert
strong nonlinearities appear, where the field decay “overshoots” and is shortly reversed
by feedback from modes, overshoots again etc.
(IV) Fluctuation dominated regime: nonperturbatively large densities ∼ 1/ λ and
quasistationary evolution. When the system evolves in time, more and more energy
is converted into fluctuations. The description in terms of O(λ 2) evolution equations
break down at
t ≃ t ′ = tnonpert : F ⊥(t,t ′; p0 ) ∼ O(N 0 λ −1 ) .
 (248)
Transverse and longitudinal modes have grown to O(N 0 λ −1 ) in a wide momentum
range for times t & tnonpert . This corresponds to nonperturbatively large particle number
densities n⊥ (p) and nk(p) inversely proportional to the coupling. Because of this para-
metric dependence there are leading contributions to the dynamics coming from all loop
orders. As a consequence, shortly after tnonpert a comparably slow, quasistationary evolu-
tion sets. It is important to note that descriptions based on a finite-order loop expansion
of the 2PI effective action cannot be applied. To describe this regime it is crucial to
employ a nonperturbative approximation as provided by the 2PI 1/N expansion at NLO.
In order to discuss this in more detail, we consider the self–energies appearing in
the evolution equations (138) for F (t,t ′; p) and ρ (t,t ′; p). From the NLO expressions
⊥ ⊥given in (184) and (185) one finds for ΣF :
⊥ΣF ⊥ (t,t ′; p)
 = −
 3N
 λ
 Z
 (2π d3 q ) 3
 n
 (249)
Introduction to Nonequilibrium Quantum Field Theory
 86
1
IF (t,t ′; q)F ⊥ (t,t ′; p − q) − 4
 Iρ (t,t ′ ; q)ρ⊥ (t,t ′; p − q)
+ P F (t,t ′; q)F ⊥ (t,t ′; p − q) − 4
 1
 P ρ (t,t ′ ; q)ρ⊥ (t,t ′; p − q)o
 .
The functions I F,ρ and P F,ρ contain the summation of an infinite number of “chain”
graphs, where each additional element adds another loop to the graph (cf. also the figures
in Sec. 2.4.1). We will argue in the following that each loop order of this infinite number
of graphs contributes with the same order in the coupling λ .
According to (181) and (182) the sum of “chain” graphs described by I F,ρ can be
iteratively generated by the relations
IF (t,t ′
; q) = − 3N
 λ
 ΠF (t,t ′; q) +
 3N λ Z
0
t dt ′′
 Iρ (t,t ′′; q)ΠF (t ′′,t ′; q)
−
 3N λ
 Z0
t′
 dt ′′ I F (t,t ′′; q)Πρ (t ′′,t ′ ; q),
 (250)
Iρ (t,t ′; q) = − 3N
 λ
 Πρ (t,t ′; q) +
 3N λ Z
t ′
 t dt ′′
 I ρ (t,t ′′; q)Πρ (t ′′,t ′ ; q) ,
 (251)
and corresponding expressions for P F,ρ (cf. Eqs. (186) and (187)). Here the “chain
elements” ΠF,ρ are given by
ΠF (t,t ′; q) = −
 2 1
 Z
 (2π d
3 k
 )3 (
F k
 (t,t ′; q − k)F k (t,t ′ ; k)
+(N − 1)F ⊥ (t,t ′; q − k)F ⊥ (t,t ′; k) − 1 4
 h
ρk (t,t ′ ; q − k)
ρk(t,t ′ ; k) + (N − 1)ρ⊥ (t,t ′′ ; q − k)ρ⊥ (t,t ′′ ; k)i
)
,
 (252)
Πρ (t,t ′
; q) = −
 Z
 (2π d 3 k )3 n
 F k
(t,t ′; q − k)ρk(t,t ′; k)
+(N − 1)F ⊥ (t,t ′; q − k)ρ⊥(t,t ′ ; k)o
 .
 (253)
(l)
Denoting a given loop order l by I we can write
F∞
I F (t,t ′
; q) =
 ∑ I F
 (l)
(t,t ′; q) ,
 (254)
l=1
(1)
 λ
I F (t,t ′; q) = −
 3N
 ΠF (t,t ′; q) ,
I F (2)
 (t,t ′
; q) = −
  3N
 λ
 2 Z0
t
 dt ′′ Πρ (t,t ′′; q)ΠF (t ′′ ,t ′; q)
...
 +
  3N
 λ
 2 Z0
t ′
 dt ′′ ΠF (t,t ′′; q)Πρ (t ′′,t ′ ; q),
Introduction to Nonequilibrium Quantum Field Theory
 87
and similarly for Iρ = ∑∞
 l=1 Iρ (l)
 . Concentrating on the transverse modes, one observes
from (252) and (253) that
ΠF (t,t ′; 0) ∼ O(N λ −2) ,
 Πρ (t,t ′; 0) ∼ O(N λ −1 ) .
 (255)
Inserting this into (250) or (254) for IF , and into the corresponding expression (251) for
I ρ , one finds
IF (l)
(t,t ′ ; 0) ∼ O(N 0λ −1 ) , I ρ (l)
(t,t ′; 0) ∼ O(N0 λ 0 ) ,
 (256)
irrespective of the loop order l. Here it is important to note that averaged over the rapid
oscillation time the spectral functions are of order one:
ρ⊥(t,t ′; p) ∼ ρk (t,t ′; p) ∼ O(N 0 λ 0) .
 (257)
As a consequence, with (257) the self–energy (250) receives leading contributions in the
coupling from all loop orders. Since these contributions are proportional N −1 λ 0 , they
are subleading in 1/N. In particular, the expansion in powers of 1/N employed here
remains a valid nonperturbative approximation scheme. It is interesting to note that the
2PI 1/N expansion to NLO can be understood as a “three-loop” approximation with an
effective vertex. This is denoted schematically below for the graphs appearing at NLO:
b
 d
d
 c
b d
b a
 d
 c
a b
 a c
 b a
b
 aa
 c
c
 d
a
 b
 a
 b
 a
 b
a
 b
 a
 b
 a
 b
c
 d
The effect of the nonperturbatively large densities is taken into account by a self-
consistent vertex correction as indicated by the diagrammatic equation for the four-
vertex. The question of how self-consistent vertex corrections can be systematically
described for cases where no 1/N expansion is applicable will be treated in Sec. 6. We
finally note that for times t & tnonpert the evolution equations are no longer characterized
by the effective time-locality in the sense described above such that the “memory
expansion” cannot be used for an approximate description at late times.
5. CLASSICAL ASPECTS OF NONEQUILIBRIUM QUANTUM
FIELDS: PRECISION TESTS
It is an important question to what extend nonequilibrium quantum field theory can be
approximated by classical statistical field theory. It is a frequently employed strategy
in the literature to consider nonequilibrium classical dynamics instead of quantum dy-
namics since the former can be simulated numerically up to controlled statistical errors.
Classical statistical field theory indeed gives important insights when the number of field
quanta per mode is sufficiently large such that quantum fluctuations are suppressed com-
pared to statistical fluctuations. We will derive below a sufficient condition for the va-
lidity of classical approximations to nonequilibrium dynamics. The description in terms
Introduction to Nonequilibrium Quantum Field Theory
 88
of spectral and statistical correlation functions as introduced in Sec. 3.4.1 is particu-
larly suitable for comparisons since these correlation functions possess a well-defined
classical limit. However, classical Rayleigh-Jeans divergences and the lack of genuine
quantum effects — such as the approach to quantum thermal equilibrium characterized
by Bose-Einstein or Fermi-Dirac statistics — limit the use of classical statistical field
theory. To find out its use and its limitations we perform below direct comparisons of
nonequilibrium classical and quantum evolution for same initial conditions. One finds
that classical methods can give an accurate description of quantum dynamics for the case
of large enough initial occupation numbers and not too late times, before the approach to
quantum thermal equilibrium sets in. Classical approaches are unsuitable, in particular,
to determine thermalization rates.
Classical methods have been extensively used in the past to rule out “candidates”
for approximation schemes applied to nonequilibrium quantum field theory. Approx-
imations that fail to describe classical nonequilibrium dynamics should be in general
discarded also for the quantum case. If the dynamics is formulated in terms of correla-
tion functions then approximation schemes for the quantum evolution can be straight-
forwardly implemented as well for the respective classical theory. For instance, the 2PI
1/N-expansion introduced in Sec. 2.4 can be equally well implemented in the classi-
cal as in the quantum theory. Therefore, in the classical statistical approach one can
compare NLO results with results including all orders in 1/N. This gives a rigorous an-
swer to the question of what happens at NNLO or beyond in this case. In particular, for
increasing occupation numbers per mode the classical and the quantum evolution can
be shown to approach each other if the same initial conditions are applied and for not
too late times. For sufficiently high particle number densities one can therefore strictly
verify how rapidly the 1/N series converges for far-from-equilibrium dynamics! The
possibility of precision tests are an important aspect of classical statistical field theory
methods.
5.1. Exact classical time-evolution equations
In this section we define the basic classical correlation functions and derive exact
dynamical equations for them. The evolution equations for the classical spectral function
and the classical statistical propagator will turn out to be of the same form as Eqs. (138)
for the corresponding quantum correlators — with the only difference that the quantum
self-energies are replaced by the respective classical ones. The comparison between
quantum and classical evolution can be very clearly discussed using the the language
of correlation functions, since the quantum spectral function ρ (x, y) and the statistical
propagator F(x, y) of Eq. (116) both have a well-defined classical equivalent.
We consider the classical N-component scalar field φa with action (13). The classical
field equation of motion is then given by

 x + m2
 +
 6N
 λ
 φb (x)φb(x)
 φa(x) = 0 ,
 (258)
whose solution requires specification of the initial conditions φa (0, x) = φa (x) and
Introduction to Nonequilibrium Quantum Field Theory
 89
πa (0, x) = πa (x) with πa (x) ≡ ∂x0 φa (x). We define the macroscopic or average classical
field:
φa,cl (x) = hφa (x)icl ≡
 Z
 Dπ Dφ W [π , φ ]φa (x) .
 (259)
Here W [π , φ ] denotes the normalized probability functional at initial time. The measure
indicates integration over classical phase-space:
Z
 Dπ Dφ =
 Z a=1 ∏ N
 ∏dπa x
 (x)dφa(x) .
 (260)
Here the theory will be defined on a spatial lattice to regulate the Rayleigh-Jeans diver-
gence of classical statistical field theory. The connected classical statistical propagator
F ab,cl (x, y) is defined by
F ab,cl(x, y) + φa,cl (x)φb,cl (y) = hφa (x)φb (y)icl ≡
 Z
 Dπ Dφ W [π , φ ]φa(x)φb (y) . (261)
The classical equivalent of the quantum spectral function is obtained by replacing −i
times the commutator with the Poisson bracket:31
ρab,cl (x, y) = −h {φa(x), φb(y)}PoissonBracket icl .
 (263)
As a consequence, one finds the equal-time relations for the classical spectral function:
ρab,cl(x, y)|x0 =y0 = 0, ∂x0 ρab,cl (x, y)|x 0 =y0 = δab δ (x − y) .
 (264)
Though their origin is different, note that they are in complete correspondence with the
respective quantum relations (117).
The evolution equations for the classical statistical correlators can be obtained by
functional methods in a similar way as for the quantum theory. In order not to be too
redundant, we employ here a different approach starting from the differential equation
for the free correlators. We will choose here W [π , φ ] to be invariant under the O(N)
symmetry with φa ≡ 0 such that
F ab,cl (x, y) = F cl (x, y)δab ,
 (265)
and equivalently for ρab,cl (x, y).32 We will discuss φa 6= 0 below. The unperturbed spec-
tral function is a solution of the homogeneous equation33

 x + m2
 ρ0 (x, y) = 0 ,
 (266)
31
 Recall that the Poisson bracket with respect to the initial fields is
{A(x), B(y)}PoissonBracket = a=1
 ∑ N Z
 dd z
 
 δ δ φa A(x) (z) δ δπ B(y)
 a (z) −
 δ δ πa A(x) (z) δ δ φa B(y)
 (z)
 
 .
 (262)
33 32
 form Z0 An ≡ for R explicit Dπ W Dφ [π , W φ example ].
 [π , φ ]. for We W emphasize [π , φ ] would that be none W [π of , φ the ] =following Z0 −1 exp R
 derivations dd x π 2 +will (∇φmake ) 2 + m2 use φ 2 of 
 /2 a specific
 where
This is the case both in the quantum and the classical theory.
Introduction to Nonequilibrium Quantum Field Theory
 90
with initial conditions determined by the equal-time canonical relations (264). Let F 0
denote the solution of the unperturbed homogeneous problem
with initial conditions determined 
x by + the m2 initial 
 F 0 (x, probability y) = 0 ,
 functional. The derivation (267)
 of
the time evolution equations of classical correlation functions is conveniently formulated
by introducing two additional two-point functions: the classical retarded and advanced
Green functions, which are related to the classical spectral function ρcl by
GR cl (x, y) = Θ(x0 − y0)ρcl(x, y) = GA cl (y, x) .
 (268)
The classical retarded self-energy ΣR,cl (x, y) can be defined as the difference between
the full and free inverse retarded Green functions
where the free retarded ΣR,cl Green (x, y) function
 = GR cl  −1
 (x, y) − GR 0
 −1
 (x, y) ,
 (269)
GR 0 (x, y) = Θ(x0 − y0 ) ρ0 (x, y) ,
solves the inhomogeneous equation
(270)
with retarded boundary conditions. 
x + m2In 
 GR the 0 (x, same y) = way δ d+1 we (x − define y) ,
 the advanced self-energy
 (271)
ΣA,cl. Retarded (advanced) Green functions and self-energies vanish when x0 < y0
(x0 > y0 ). With these definitions we can rewrite (269) and the respective equation for
the advanced Green function as:
GR cl = G0 R − G0 R · ΣR,cl · Gcl R ,
GA cl = GA 0 − GA 0 · ΣA,cl · GA cl ,
where we use a compact notation
(272)
(273)
A · B =
 Z
 dd+1 z A(x, z)B(z, y) .
 (274)
A combination of Eqs. (272) and (273) gives the Schwinger-Dyson equation for the
classical spectral function ρ cl = G R cl − GA cl with
ρcl = ρ0 − GR 0 · ΣR,cl · GR cl + GA 0 · ΣA,cl · GAcl .
 (275)
There is a similar identity for the classical statistical propagator F cl , defined in (261).
Using the definitions (272) and (273) one can write the following identity for the
statistical function
F cl = F 0 − G0 R
 · h
 ΣR,cl − Gcl
 R −1
 i
 · F cl − F 0 · h
ΣA,cl + G0
 A −1
 i
 · GA cl .
 (276)
Introduction to Nonequilibrium Quantum Field Theory
 91
We can now define a classical statistical self-energy as
ΣF,cl = −G R cl
 −1
 · F cl · GA cl
−1
 + GR 0
 −1
 · F 0 · GA 0
 −1
 ,
 (277)
and find the Schwinger-Dyson equation for the statistical propagator:
F cl = F 0 − GR 0 · ΣR,cl · F cl − F 0 · ΣA,cl · GA cl − GR 0 · ΣF,cl · GA cl .
 (278)
Acting with ( + m2) on (275) and (278) brings the classical Schwinger-Dyson equa-
tions in a form which is more suitable for initial-value problems. We make the retarded
nature of ΣR,cl manifest by writing
ΣR,cl(x, y) = Σcl (0)
(x)δ d+1 (x − y) + Θ(x0 − y0 )Σρ ,cl(x, y) ,
 (279)
and similarly for ΣA,cl (x, y). The spectral component of the classical self-energy is
Σρ ,cl (x, y) = ΣR,cl(x, y) − ΣA,cl(x, y) = −Σρ ,cl (y, x). After properly taking into account
all Θ–functions as well as (266), (271) and (267), one finds for the exact time evolution
equations for ρcl and F cl :

x + M cl 2
 (x)
 ρcl(x, y) = −
 Z
y0
 x0
 x0
dz Σρ ,cl(x, z)ρcl (z, y) ,
 (280)

 x + Mcl 2
 (x)
 F cl (x, y) = −
 Z
0
y0
dz Σρ ,cl(x, z)F cl(z, y)
+
 Z0
 dz ΣF,cl(x, z)ρcl (z, y) ,
 (281)
where we use the abbreviated notation
 2 Rt1
 t2
 dz 2 ≡
 R t1
 t2
 (0)
 dz0
 R−∞ ∞
 d
d z
 and
Mcl(x) = m + Σcl (x).
 (282)
We emphasize that the form of these classical time evolution equations is the same as for
the quantum evolution described by (138). If the initial conditions are chosen to be the
same then the only difference between classical and quantum theory originates from the
self-energies entering the evolution equations. As a direct consequence, one concludes
that LO large-N or any Gaussian/Hartree type approximation exhibits purely classical
dynamics: since the quantum self-energies ΣF and Σρ as well as the classical ΣF,cl and
Σρ ,cl vanish for these approximations the evolution equations derived from the quantum
theory are identical to those derived from the classical theory. The tadpole contributions
Σ(0) and Σcl (0)
 entail no difference if the initial conditions are chosen to be the same. In the
following we consider approximations with non-vanishing spectral and statistical self-
energies in order to discuss the differences between classical and quantum dynamics.
5.2. Classicality condition
The classical self-energies Σρ ,cl, ΣF,cl(x, z) and Σcl (0)
 could be approximated by pertur-
bation theory to a given order in the coupling λ . However, as for the case of the quantum
Introduction to Nonequilibrium Quantum Field Theory
 92
evolution the perturbative classical self-energies lead to a secular time-evolution and fail
to provide a reliable description for the classical nonequilibrium dynamics. The ana-
lytical description can, however, be based on the same 2PI summation techniques as
introduced for the quantum theory above. We will only state here the result of the re-
spective calculation within classical statistical field theory, since we will then discuss
how the same result can be directly obtained from the classical limit of the known quan-
tum self-energies.
One finds for the classical statistical O(N) model employing the 2PI 1/N expansion
to next-to-leading order:
2 2 N + 2
M cl (x) = m + λ
 F cl (x, x) ,
6N
(NLO)
 λ
ΣF,cl (x, y) = −
 3N
 F cl (x, y)I F,cl (x, y) ,
 (283)
Σρ (NLO)
 ,cl (x, y) = −
 3N
 λ 
F cl (x, y)I ρ ,cl (x, y) + ρcl(x, y)IF,cl(x, y)
 ,
with the classical summation functions
λ 2
I F,cl (x, y) =
 F cl
 (x, y)
6−
 λ
 6
 Z
 dd
 z
 
 Z
 
 0
 x
 0
 dz0
 Iρ ,cl (x, z)F cl 2
 (z, y) − 2 Z
 0
 y
0
 dz0
 IF,cl(x, z)F cl (z, y)ρcl(z, y)
 
 
 ,
λ
 
  λ
 x
0
 
 
I ρ ,cl (x, y) =
 3
 F cl (x, y)ρcl(x, y) −
 3
 Z
 dd z
Z dz0 Iρ ,cl(x, z)F cl(z, y)ρcl(z, y) .
 (284)
y0
At this stage we can compare the result from the 2PI 1/N expansion for the quantum
theory, as discussed in Sec. 2.4, with the resummed classical expressions obtained
here. One observes that the classical self-energies (283)—(284) are obtained from the
corresponding quantum expressions (178)—(182) by dropping the terms with a product
of two spectral functions ρ compared to a product of two statistical propagators F.
Accordingly, the classical self-energies are obtained from the respective quantum ones
as
ΣF,cl = ΣF (F 2 ≫ ρ 2 ) , Σρ ,cl = Σρ (F 2 ≫ ρ 2 ) .
 (285)
The same analysis can be done employing the 2PI loop expansion. For the follow-
ing analytical discussion we consider the spatially homogeneous case and employ the
Fourier modes ΣF,cl(t,t ′ ; p) and Σρ ,cl (t,t ′; p). For the classical statistical O(N) symmet-
ric theory one obtains to two-loop order the self-energies:
ΣF,cl (2loop)
 (t,t ′; p)
 =
 −λ 2
 N 18N + 2
 2
 Z
q,k
 F cl (t,t ′; p − q − k)F cl(t,t ′; q)F cl(t,t ′ ; k) ,
Σρ (2loop)
 ,cl (t,t ′; p)
 =
 −λ 2
 N 6N + 2
 2
 Z
q,k
 ρcl(t,t ′ ; p − q − k)F cl (t,t ′; q)F cl (t,t ′; k) . (286)
Introduction to Nonequilibrium Quantum Field Theory
 93
Let us directly confront this with the respective two-loop self-energies for the respective
quantum O(N) theory:
ΣF
 (2loop)
 (t,t ′; p)
 =
 −λ 2
 N 18N + 2
 2
 Z
 q,k
 F(t,t ′; p − q − k)

F(t,t ′
; q)F(t,t ′
; k)− 4
 3
 ρ (t,t ′
; q)ρ (t,t ′
; k)
 ,
Σρ
 (2loop)
 (t,t ′; p) = −λ 2
 N
 6N
 +
 2
 2
 Z
 q,k
 ρ (t,t ′; p − q − k)

F(t,t ′
; q)F(t,t ′
; k)− 12
 1
 ρ (t,t ′
 ; q) ρ (t,t ′
 ; k)
 .
 (287)
From this one infers a sufficient condition for classical evolution:
|F(t,t ′; q)F(t,t ′; k)| ≫
3
4
|ρ (t,t ′ ; q)ρ (t,t ′; k)|
(288)
This condition has to be fulfilled for all times and all momenta in order to ensure that
classical and quantum evolution agree. However, we will observe below that it can be
typically only achieved for a limited range of time and/or momenta.
One expects that the classical description becomes a reliable approximation for the
quantum theory if the number of field quanta in each mode is sufficiently high. The
classicality condition (288) entails the justification of this expectation. In order to il-
lustrate the condition in terms of a more intuitive picture of occupation numbers, we
employ the free-field theory type form of the spectral function and statistical propagator
given in Eq. (211). From this one obtains the following estimates for the time-averaged
correlators:
F 2(t,t ′ ; p) ≡
 2π
 ω p
 Zt−2π t
 dt /ωp
 ′ F 2 (t,t ′; p) →
 (np (t) 2ω + p 2 (t)
 1/2)2
 ,
1
ρ 2(t,t ′ ; p) →
 2ωp 2 (t)
 .
 (289)
Inserting these estimates in (288) for equal momenta yields

n p(t) +
 2
 1 
2
 ≫
 3
 4
 or
 np (t) ≫ 0.37 .
 (290)
This limit agrees rather well with what is obtained for the case of thermal equilibrium.
For a Bose-Einstein distributed particle number nBE = (eω /T − 1)−1 with temperature
T one finds nBE ( ω = T ) = 0.58, below which deviations from the classical thermal
distribution become sizeable. We emphasize that in thermal equilibrium the fluctuation-
dissipation relation (129) ensures |F (eq) (ω , p)| ≫ |ρ (eq)(ω , p)| for high-temperature
modes T ≫ ω :
F (eq) (ω , p) = −i
nBE ( ω ) +
 2
 1  ρ (eq)
 (ω , p) T ≫ω
 ≃ −i T
 ω
 ρ (eq) (ω , p) .
 (291)
Introduction to Nonequilibrium Quantum Field Theory
 94
The nonequilibrium estimate (290) as well as the equilibrium result (291) leads one to
expect that the quantum evolution described by F and ρ is well approximated by the
classical evolution in terms of F cl and ρcl for large initial occupation numbers np (t = 0).
However, note that the nonequilibrium classicality condition (288) is typically not ful-
filled at all times, since the unequal-time correlator F(t,t ′; p) can oscillate around zero
with a phase difference to the oscillations of ρ (t,t ′ ; p) (cf. also the free-field expres-
sions (211)). In particular, we will find that characteristic quantities for the quantum
late-time behavior such as thermalization times are very badly approximated by the re-
spective classical estimates. Nevertheless, for time-averaged quantities and not too late
times t ≪ teq, i.e. before the approach to quantum thermal equilibrium sets in, the clas-
sical evolution can give an accurate description for sufficiently high initial occupation
numbers as will be demonstrated below.
A similar analysis can be performed in the presence of a non-vanishing classical
macroscopic field φa,cl (x) defined in (259). For instance, to NLO in the 2PI 1/N ex-
pansion the classical macroscopic field is described by the evolution equation

x +
 6N λ φcl
 2
 (x)
 δab + M cl,ab 2
 (x; φcl ≡ 0, F cl )
 φcl,b (x)
= −
 Z0
x0
dy Σρ cl ,ab (x, y; φcl ≡ 0, F cl, ρ cl) φcl,b(y) ,
 (292)
where Mcl,ab
 2 (x; φcl
 ≡ 0, F cl
 ) is given by (146) with the replacement F → F cl
 , and
Σρ cl
 ,ab(x, y; φcl ≡ 0, F cl, ρcl) is obtained from the respective quantum self-energy
Σρ ,ab(x, y; φ ≡ 0, F, ρ ) given in (185) by employing (285). This can be directly compared
to the corresponding evolution equation (189) for the case of the quantum field theory.
5.3. Precision tests and the role of quantum corrections
The nonequilibrium evolution of classical correlation functions in the O(N) model
can be obtained numerically up to controlled statistical errors. Initial conditions for
the nonequilibrium evolution are determined from a probability functional on classi-
cal phase-space. The subsequent time evolution is solved numerically using the classical
equation of motion for the field (258). The results presented below have been obtained
from sampling 50000-80000 independent initial conditions to approximate the exact
evolution of correlation functions. The statistical propagator is constructed from these
individual runs according to (261) and using (263) for the spectral function. Since these
results include all orders in 1/N, they can be used for a precision test of the 2PI 1/N
expansion implemented in classical statistical field theory. We emphasize that this com-
pares two very different calculational procedures: the results from the simulation involve
thousands of individual runs from which the correlators are constructed, while the cor-
responding results employing the 2PI 1/N expansion involve only a single run solving
directly the evolution equation for the correlators. The accuracy of the simulations man-
ifests itself also in the fact that the time-reversal invariant dynamics can be explicitly
reversed in practice for not too late times. The close agreement between full and 2PI
Introduction to Nonequilibrium Quantum Field Theory
 95
0.6
)
0 0.3
=p;0, 0
t(F−0.3
N=20
N=10
N=2
NLOexactclassical
classical
0.5
0.4
γ
0.3
0.2
0.1
exactNLONLOclassical
classical
quantum
0
−0.6
 0
 0.1
 0.2
 0.3
 0.4
 0.5
0
 5
 10
 t
 15
 20
 1/N
FIGURE 16. Left: Unequal-time two-point function F(t, 0; p = 0) at zero momentum for N = 2, 10, 20.
The full lines show results from the NLO classical evolution and the dashed lines from the exact classical
evolution (MC). One observes a convergence of classical NLO and exact results already for moderate
values of N. Right: Damping rate extracted from F(t, 0; p = 0) as a function of 1/N. Open symbols
represent NLO and exact classical evolution. The quantum results are shown with full symbols for
comparison. The initial conditions are characterized by low occupation numbers so that quantum effects
become sizeable. One observes that in the quantum theory the damping rate is reduced compared to the
classical theory. (All in units of mR.)
NLO results, which will be observed below, is therefore also indicative of the numerical
precision of the latter.
We consider a system that is invariant under space translations and work in momen-
tum space. We choose a Gaussian initial state such that a specification of the initial
two-point functions is sufficient. As mentioned above, the classical spectral function at
initial time is completely determined from the equal-time relations (264). For the sta-
tistical propagator we take F(0, 0; p) = [n p(0) + 12 ]/ω p , with the initial particle number
representing a peaked “tsunami” (cf. Sect. 4.1.1).
The initial mode energy is given by ω p = (p2 + Mcl 2 )1/2 where Mcl is the one-loop
renormalized mass in presence of the nonequilibrium medium, determined from the one-
loop “gap equation” for Mcl in Eq. (283). As a renormalization condition we choose the
one-loop renormalized mass in vacuum mR ≡ M|n(0)=0 = 1 as our dimensionful scale.
The results shown below are obtained using a fixed coupling constant λ /m2 R = 30.
On the left of Fig. 16 the classical statistical propagator F cl (t, 0; p = 0) is presented
for three values of N. All other parameters are kept constant. The figure compares the
time evolution using the 2PI 1/N expansion to NLO and the full Monte Carlo (MC)
calculation. One observes that the approximate time evolution of the correlation function
shows a rather good agreement with the exact result even for small values of N (note
that the effective four-point coupling is strong, λ /6N = 2.5m2 R for N = 2). For N = 20
the exact and NLO evolution can hardly be distinguished. A very sensitive quantity for
comparisons is the damping rate γ , which is obtained from an exponential fit to the
envelope of F cl (t, 0; p = 0). The systematic convergence of the NLO and the Monte
Carlo result as a function of 1/N can be observed from the right graph of Fig. 16. The
quantitatively accurate description of far-from-equilibrium processes within the NLO
Introduction to Nonequilibrium Quantum Field Theory
 96
approximation of the 2PI effective action is manifest.
The right graph of Fig. 16 also shows the damping rate from the quantum evolution,
using the same initial conditions and parameters. One observes that the damping in
the quantum theory differs and, in particular, is reduced compared to the classical
result. The effective loss of details about the initial conditions takes more time for the
quantum system than for the corresponding classical one. In the limit N → ∞ damping
of the unequal-time correlation function goes to zero since the nonlocal part of the self-
energies vanishes identically at LO large-N and scattering is absent. In this limit there
is no difference between evolution in a quantum and classical statistical field theory.
(Cf. Sec. 4.1.1 and the discussion above.)
For finite N scattering is present and quantum and classical evolution differ in general.
However, as discussed in Sec. 5.2, the classical field approximation may be expected
to become a reliable description for the quantum theory if the number of field quanta
in each field mode is sufficiently high. We observe that increasing the initial particle
number density leads to a convergence of quantum and classical time evolution at not
too late times. In Fig. 17 (left) the time evolution of the equal-time correlation function
F(t,t; p) is shown for several momenta p and N = 10. Here the integrated particle
n density p=2pts (0) R
 2 d ≃ π p n 0.35 p (0)/Mcl and a = slightly 1.2 is six larger times value as at high this as momentum in Fig. 16. of At about p = 2pts ≃ 0.5 one for finds
 late
times. For these initial conditions the estimate (290) for the classicality condition (288)
is therefore approximately valid up to momenta p ≃ 2pts , and one indeed observes from
the left of Fig. 17 a rather good agreement of quantum and classical evolution in this
range. For an estimate of the NLO truncation error we also give the full (MC) result for
N = 10 showing a quantitative agreement with the classical NLO evolution both at early
and later times.
5.3.1. Classical equilibration and quantum thermal equilibrium
From the left of Fig. 17 one observes that the initially highly occupied “tsunami”
modes “decay” as time proceeds and the low momentum modes become more and more
populated. At late times the classical theory and the quantum theory approach their
respective equilibrium distributions. Since classical and quantum thermal equilibrium
are distinct, the classical and quantum time evolutions have to deviate at sufficiently late
times. Figure 17 shows the time dependent inverse slope parameter
T (t, p) ≡ −n p(t)[n p(t) + 1]
 
 dn d ε p
 p 
−1
 ,
 (293)
which has been introduced in Sec. 4.1.2 to study the approach to quantum thermal
equilibrium.34 It employs the effective particle number n p (t) defined in (201)35 and
mode energy ε p(t) given by (203). Initially one observes a very different behavior of
34
 Note that dLog(n−1
 p (t) + 1)/d ε p (t) = T
 −1(t, p).
35
 Here we employ Q(t,t ′
 ; p) = 0, cf. Sec. 4.1.2.
Introduction to Nonequilibrium Quantum Field Theory
97
)
p;t,t(F1
exact classical
NLO classical
NLO quantum
p=0
p ~ pts
12
10
)
p,t(T8
6
plow
NLONLOquantum
classical
p ~ 2pts
 4
phigh
0
0
 50
 100
 150
 0
 50
 100
 150
 200
 250
t
 t
FIGURE 17. Left: Nonequilibrium evolution of the equal-time two-point function F(t,t; p) for N = 10
for various momenta p. One observes a good agreement between the exact MC (dashed) and the NLO
classical result (full). The quantum evolution is shown with dotted lines. The integrated initial particle
density is six times as high as in Fig. 16. Right: A very sensitive quantity to study deviations is the
time dependent inverse slope T (t, p) defined in the text. When quantum thermal equilibrium with a Bose-
Einstein distributed particle number is approached all modes get equal T (t, p) = T eq, as can be observed
to high accuracy for the quantum evolution. For classical thermal equilibrium the defined inverse slope
remains momentum dependent.
T (t, p) for the low and high momentum modes, indicating that the system is far from
equilibrium. The quantum evolution approaches quantum thermal equilibrium with a
momentum independent inverse slope T eq = 4.7 mR to high accuracy. In contrast, in the
classical limit the slope parameter remains momentum dependent since the classical
dynamics does of course not reach a Bose-Einstein distribution.
To see this in more detail we note that for a Bose-Einstein distribution, nBE(ε p) =
1/[exp(ε p/T eq ) − 1], the inverse slope (293) is independent of the mode energies and
equal to the temperature T eq . During the nonequilibrium evolution effective thermaliza-
tion can therefore be observed if T (t, p) becomes time and momentum independent,
T (t, p) → T eq . This is indeed seen on the left of Fig. 17 for the quantum system. If the
system is approaching classical equilibrium at some temperature T cl and is weakly cou-
pled, the following behavior is expected. From the definition (201) of n p(t) in terms in
two-point functions, we expect to find approximately
i.e. a remaining momentum dependence T (t, p) → T with cl 1 T − (t, ε p 2 /T p) cl < 2 
 T ,
 (t, p′) if ε p > ε ′ p . Indeed, (294)
 this
is what one observes for the classical field theory result in Fig. 17.
For a classical theory a very simple test for effective equilibration is available. An
exact criterion can be obtained from the classical counterpart of the “KMS” condition for
thermal equilibrium discussed in Sec. 3.4.2. In coordinate space the classical equilibrium
“KMS” condition reads
1 ∂ (eq)
 (eq)
T cl ∂ x
0
 F cl (x − y) = −ρcl (x − y),
 (295)
Introduction to Nonequilibrium Quantum Field Theory
 98
9.5
9
8.5
8
)
p , 7.5
t(l c 7
T6.5
6
5.5
5
0
 20 40 60 80 100
 4000 8000 12000 16000
t
FIGURE 18. Nonequilibrium time evolution in 1 + 1 dimensions in the classical limit of the three-
loop approximation of the 2PI effective action for one scalar field, N = 1. Shown is the effective mode
temperature T cl (t, p), defined in the text, with initial T cl (0, p)/mR = 5 for all p. The classical field theory
approaches classical equilibrium, T cl (t, p) → T cl , after a very long time.
and in momentum space
(eq)
 0 (eq)
 T cl
F cl (k) = −incl (k )ρcl (k),
 ncl(k0 ) =
 k 0
 .
 (296)
Differentiating Eq. (295) with respect to y0 at x0 = y0 = t gives
1 ∂ ∂ (eq)
 ∂ (eq)
T cl ∂ y0 ∂ x0 F cl
 (x − y)
 x0 =y0 =t
 = −
 ∂ y0 ρcl
 (x − y)
 x0=y0 =t
 .
 (297)
Combining this KMS relation with the equal-time condition (264) for the spectral
function leads to
∂ t ∂ t ′ F cl (eq)
 (t,t ′; x − y)|t=t′ = T clδ (x − y).
 (298)
In terms of the fluctuating classical fields πa(eq)
 (x) ≡ ∂x0 φa (x) this is of course the well-
known equilibrium relation hπa (t, x)πb(t, y)icl = T clδ (x − y)δab. Out of equilibrium
one can define an effective classical mode temperature
T cl (t, p) = ∂t ∂t ′ F cl(t,t ′ ; p)| t=t ′ ,
 (299)
and effective classical equilibration is observed if T cl (t, p) becomes time and momentum
independent, T cl (t, p) → T cl .
Apart from the 2PI 1/N expansion the late-time behavior can also be studied from the
2PI loop expansion. For the quantum theory this has been demonstrated in Sec. 4.1.4 for
the three-loop approximation of the 2PI effective action (cf. Fig. 6, left). The equivalent
approximation in the classical theory is given by the evolution equations (280) and (281)
with the two-loop self-energies (286). In order to demonstrate this, we will use the loop
approximation to verify explicitly the above statements about classical equilibration.
Introduction to Nonequilibrium Quantum Field Theory
 99
In Fig. 18 the nonequilibrium evolution of the effective “mode temperature” T cl (t, p)
is shown for various momentum modes in the (1 + 1)-dimensional classical scalar field
theory for N = 1. The equations are solved by a lattice discretization with spatial lattice
spacing mR a = 0.4, time step a0/a = 0.2, and Ns = 24 sites. Without loss of generality
we use λ /m2 R = 1. For the initial ensemble we take F cl(0, 0; p) = T 0/(p2 + m2 R ) and
∂ t ∂ t ′ F cl (t,t ′; p)|t=t ′=0 = T 0 with T 0/mR = 5. We have observed that at sufficiently late
times the contributions from early times to the dynamics are effectively suppressed.
This fact has been employed in Fig. 18 to reach the very late times. One sees that
at sufficiently late times the system relaxes towards classical equilibrium with a final
temperature T cl /mR ≈ 5.5. For the zero-momentum mode we find an exponential late-
time relaxation of T cl(t, p = 0) towards T cl with a small rate (∼ 2 × 10−4mR). It should
be emphasized that typical classical equilibration times are substantially larger than the
times required to approach thermal equilibrium in the respective quantum theory. In
contrast to the quantum theory statements about equilibration times are, however, in
general not insensitive to the employed momentum regularization for the classical theory
because of the classical Rayleigh-Jeans divergences.
6. N-PARTICLE IRREDUCIBLE GENERATING
FUNCTIONALS II: EQUIVALENCE HIERARCHY
To understand the success and, more importantly, the limitations of expansion schemes
based on the 2PI effective action we consider in this section n-particle irreducible (nPI)
effective actions for n > 2. Recall that the description of the 2PI effective action Γ[φ , D]
employs a self-consistently dressed one-point function φ and two-point function, which
for notational purposes we denote here by D:36 The one-point and two-point functions
are dressed by solving the equations of motion δ Γ/δ φ = 0 and δ Γ/δ D = 0 for a given
order in the (e.g. loop) expansion of Γ[φ , D] (cf. Sec. 2). However, the 2PI effective
action does not treat the higher n-point functions with n > 2 on the same footing as the
lower ones: The three- and four-point function etc. are not self-consistently dressed in
general, i.e. the corresponding proper three-vertex V 3 and four-vertex V 4 are given by
the classical ones. In contrast, the nPI effective action Γ[φ , D,V 3,V 4, . . . ,V n] provides
a dressed description for the proper vertices V 3 ,V4 , . . . ,Vn as well, with δ Γ/δ V 3 =
0, δ Γ/δ V 4 = 0, . . ., δ Γ/δ V n = 0.
The use of nPI effective actions with higher n > 2 is not entirely academic. They
are relevant in the presence of initial-time sources describing a non-Gaussian initial
density matrix for nonequilibrium evolutions (cf. the discussion in Secs. 3.2 and 3.3).
They are also known to be relevant in high-temperature gauge theories for a quantitative
description of transport coefficients in the context of kinetic theory. As an example, the
calculation of shear viscosity in a theory like QCD can be based on the inclusion of an
infinite series of 2PI “ladder” diagrams in order to recover the leading order “on-shell”
results in the gauge coupling g. Further examples where approximation schemes based
36
 G will denote the ghost propagator in gauge theories below.
Introduction to Nonequilibrium Quantum Field Theory
100
on higher nPI effective actions are relevant include critical phenomena near second-order
phase transitions. For instance, the quantitative description of the universal behavior
near the second-order phase transition of scalar φ 4 theory goes beyond a 2PI loop
expansion.37 It requires taking into account vertex corrections that start with the 4PI
effective action to four-loop order. The latter agrees with the most general nPI loop
expansion to that order, which is shown below.
The evolution equations, which are obtained by variation of the nPI effective action,
are closely related to known exact identities for correlation functions, i.e. Schwinger-
Dyson (SD) equations. Without approximations the equations of motion obtained from
an exact nPI effective action and the exact SD equations have to agree since one can
always map identities onto each other. However, in general this is no longer the case for
a given order in the loop or coupling expansion of the nPI effective action. By construc-
tion, SD equations are expressed in terms of loop diagrams including both classical and
dressed vertices, which leads to ambiguities of whether classical or dressed ones should
be employed at a given truncation order. In particular, SD equations are not closed a
priori in the sense that the equation for a given n-point function always involves infor-
mation about m-point functions with m > n. These problems are absent using effective
action techniques. In turn, the nPI results can be used to resolve ambiguities of whether
classical or dressed vertices should be employed for a given truncation of a SD equation.
For instance, in QCD the three-loop effective action result leads to evolution equations,
which are equivalent to the SD equation for the two-point function and the one-loop
three-point function if all vertices in loop-diagrams for the latter are replaced by the full
vertices at that order. As mentioned in previous sections, the “conserving” property of
using an effective action truncation can have important advantages, in particular if ap-
plied to nonequilibrium time evolution problems, where the presence of basic constants
of motion such as energy conservation is crucial.
We will derive below the 4PI effective action for a nonabelian SU (N) gauge theory
with fermions up to four-loop or O(g6 ) corrections, starting from the 2PI effective action
and doing subsequent Legendre transforms (Secs. 6.1 and 6.2). The class of models
include gauge theories such as QCD or abelian theories as QED, as well as simple scalar
field theories with cubic or quartic interactions. In Sec. 6.1.2 we derive an equivalence
hierarchy for nPI effective actions, which implies that the 4PI results to this order are
equivalent to those from the nPI effective action up to four-loop or O(g6 ) corrections
for arbitrary n > 4. We derive the non-equilibrium gauge field and fermion evolution
equations (Sec. 6.2.3), and discuss the connection to kinetic theory in Sec. 6.3.
37
 Critical phenomena can be described using the 1/N expansion of the 2PI effective action beyond
leading order. These approximations indeed resum an infinite number of loop diagrams and, for instance,
the NLO result can be rewritten as a “loop approximation in the presence of an effective four-vertex”
(cf. Secs. 2.1 and 4.3).
Introduction to Nonequilibrium Quantum Field Theory
 101
6.1. Higher effective actions
Recall that all information about the quantum theory can be obtained from the ef-
fective action, which is a generating functional for Green’s functions evaluated in the
absence of external sources, i.e. at the physical or stationary point. All functional rep-
resentations of the effective action are equivalent in the sense that they are generating
functionals for Green’s functions including all quantum/statistical fluctuations and, in
the absence of sources, have to agree by construction:38
Γ[φ ] = Γ[φ , D] = Γ[φ , D,V 3] = Γ[φ , D,V 3,V 4 ] = . . . = Γ[φ , D,V 3 ,V 4, . . . ,V n]
 (300)
for arbitrary n without further approximations. However, e.g. loop expansions of the
1PI effective action to a given order in the presence of the “background” field φ differ in
general from a loop expansion of Γ[φ , D] in the presence of φ and D. A similar statement
can be made for expansions of higher functional representations. As mentioned in
the introduction in Sec. 1.1.2, for applications it is often desirable to obtain a self-
consistently complete description, which to a given order in the expansion determines
Γ[φ , D,V 3,V4 , . . . ,Vn ] for arbitrarily high n. For this it is important to realize that there
exists an equivalence hierarchy as displayed in the introduction in Eq. (13), which is
derived in Sec. 6.1.2. For instance at three-loop order one has:
Γ(3loop)[φ ] 6= Γ(3loop)[φ , D] 6= Γ(3loop)[φ , D,V 3 ]==Γ(3loop) [φ , D,V 3 ,V 4]
 (301)
Γ(3loop) [φ , D,V 3 ,V 4, . . . ,V n] ,
for arbitrary n > 4 in the absence of sources. As a consequence, there is no differ-
ence between Γ(3loop) [φ , D,V 3] and Γ(3loop)[φ , D,V 3,V 4 ] etc. at the stationary point, and
the 3PI effective action captures already the complete answer for the self-consistent
description to this order. More explicitly, the equality reads Γ(3loop) [φ , D(φ ),V3(φ )] =
Γ(3loop) [φ , D(φ ),V 3(φ ),V4(φ )] = . . . since at the stationary point of the effective action
all n-point correlations become functions of the field expectation value φ . At four-loop
order the 4PI effective action would become relevant etc. For instance, for a theory as
quantum electrodynamics (QED) or chromodynamics (QCD) the 2PI effective action
provides a self-consistently complete description to two-loop order or39 O(g2 ): For a
sources. two-loop In approximation contrast, a self-consistently all nPI descriptions complete with result n ≥ 2 to are three-loop equivalent order in the or absence O(g4 ) re-
 of
quires at least the 3PI effective action etc. To go to much higher loop-order can become
somewhat academic from the point of view of calculational feasibility.
To present the argument we will first consider the 4PI effective action for a simple
generic scalar model with cubic and quartic interactions. The formal generalization to
38
 Of course, the nonequilibrium nPI effective action in the presence of initial-time sources representing
an initial density matrix can differ in general. However, these pose no additional complications since they
vanish identically for times different than the initial time. Cf. the discussions in Secs. 3.3 and 3.4.3.
39 Here, and throughout the paper, g means the strong gauge coupling gs
 for QCD, while it should be
The understood metric is as denoted the electric as gμν charge = g μν e for = QED. diag(1, For the power counting we take φ ∼ O(1/g) (cf. Sec. 6.1.1).
−1, −1, −1).
Introduction to Nonequilibrium Quantum Field Theory
 102
fermionic and gauge fields is straightforward, and in Sec. 6.2 the construction is done
for SU (N) gauge theories with fermions. We use here a concise notation where Latin
indices represent all field attributes, numbering real field components and their internal
as well as space-time labels, and sum/integration over repeated indices is implied. We
consider the classical action
1
 g
 g2
S[φ ] = 2
 φi iD−1
 0,i j φ
 j
 −
 3!
 V
 03,i jk φ
i φ
 j φ
k
 −
 4!
 V 04,i jkl φi φ j φk φl ,
 (302)
where we scaled out a coupling constant g for later convenience. The generating func-
tional for Green’s functions in the presence of quadratic, cubic and quartic source terms
is given by:
Z[J, R, R3 , R4 ] = exp (iW [J, R, R3, R4])
=
 Z
 1
 D φ exp ni
S[φ ] + 1
 Ji φi + 1
 2
 Ri j φi φ j
 (303)
+ 3!
 R3,i jk φi φ j φk + 4!
 R4,i jkl φiφ j φk φl o
 .
The generating functional for connected Green’s functions, W , can be used to define the
connected two-point (D), three-point (D3 ) and four-point function (D4 ) in the presence
of the sources,
δ W
= φi ,
 (304)
δ Ji
δ W
 1
δ δ W
 Ri j
 =
 1
 2
 Di j + φi φ j 
 ,
 (305)
δδ R3,i W
 jk
 =
 6
 1
 D3,i jk + Di j φk + Dki φ j + D jk φi + φi φ j φk 
 ,
 (306)
=
 (D4,i jkl + [D3,i jk φl + 3 perm.] + [Di j Dkl + 2 perm.]
δ R4,i jkl
 24
+[Di j φk φl + 5 perm.] + φi φ j φk φl ) .
 (307)
We denote the proper three-point and four-point vertices by gV3 and g2 V 4 , respectively,
and define
D3,i jkD4,i jkl==−ig Dii′ D j j ′ Dkk′ V 3,i′ j ′ k′ ,
−ig2
 Dii′ D j j ′ Dkk′ Dll ′V 4,i′ j ′ k′ l ′
+g2 (Dii′ D j j ′ Dk′u′ Dw′l Dv′ k + Dii′ D j′u′ Dk′ l D jv′ Dw′ k
+Dii ′ D j ′u′ Dk′k D jv′ Dl ′ l )V 3,i′ j′k′V 3,u′ v′ w ′ .
(308)
(309)
Introduction to Nonequilibrium Quantum Field Theory
 103
The effective action is obtained as the Legendre transform of W [J, R, R3 , R4 ]:40
δ W
 δ W
Γ[φ , D,V 3 ,V 4] = W −
 δ Ji
 Ji −
 δ Ri j
 Ri j
δ W
 δ W
−
 δ R 3,i jk
 R3,i jk −
 δ R4,i jkl
 R4,i jkl .
For vanishing sources one observes from (310) the stationarity conditions
δΓ δΓ
 δΓ
 δΓ
=
 =
 =
 = 0 ,
δφ
 δ D δ V3 δ V 4
which provide the equations of motion for φ , D, V 3 and V 4 .
(310)
(311)
6.1.1. 4PI effective action up to four-loop order corrections
Since the Legendre transforms employed in (310) can be equally performed subse-
quently, a most convenient computation of Γ[φ , D,V3,V 4 ] starts from the 2PI effective
action Γ[φ , D]. According to (28) the exact 2PI effective action can be written as:
i
 i
Γ[φ , D] = S[φ ] + Trln D−1 + Tr D−1
 0 (φ )D + Γ2[φ , D] + const ,
 (312)
2
 2
with the field-dependent inverse classical propagator
δ 2 S[φ ]
iD−1
 0 (φ ) =
 .
 (313)
δφδφ
To simplify the presentation, we use in the following a symbolic notation which sup-
presses indices and summation or integration symbols (suitably regularized). In this no-
40
 In terms of the standard one-particle irreducible effective action Γ[φ ] = W [J] − J φ the proper vertices
V 3 and V 4 are given by
δ 3 Γ[φ ]
 δ 4 Γ[φ ]
gV3 = −
 δφδφδφ
 ,
 g 2V 4 = −
 δφδφδφδφ
 .
Here it is useful to note that in terms of the connected Green’s functions Dn one has
δ 2W [J]
 δ 2 Γ[φ ]
= iD
 ,
 = i D−1 ,
δ J δ J
 δφδφ
δ 3W [J]
 δ 3 Γ[φ ]
δ Jδ J δ J
 = − D3 = −i D3
 δφδφδφ
 .
δ J δ δ 4W Jδ[J]
 J δ J
 = −i D4 = D4
 δφδφδφδφ
 δ 4 Γ[φ ]
 + 3i D5
  δφδφδφ
 δ 3
 Γ[φ ]
 2
 .
Introduction to Nonequilibrium Quantum Field Theory
 104
tation the inverse classical propagator reads
1 2 2
iD−1
 0 (φ ) = iD−1
 0 − gφ V 03 − 2 g φ V 04 ,
 (314)
and to three-loop order one has41 (cf. Sec. 2.1)
1
 i
 i
 2
Γ2 [φ , D] = − 8
 g2 D2V04 + 12
 D3 (gV 03 + g2 φ V04 )2 + 48
 g4 D4 V 04
1
 i
+ 8
 g2 D5 (gV 03 + g2 φ V 04 )2V 04 − 24
 D6 (gV 03 + g2 φ V 04 )4
for n, m = 0, . . ., 6. We emphasize + O gn (g2 that φ )m
 the | n+m=6 exact 
 ,
 φ -dependence of Γ2 [φ , D] can be written
 (315)
as a function of the combination (gV 03 + g2 φ V 04 ). In order to obtain the vertex 2PI
effective action Γ[φ , D,V 3,V4 ] from Γ[φ , D], one can exploit that the cubic and quartic
source terms ∼ R3 and ∼ R4 appearing in (303) can be conveniently combined with the
vertices gV 03 and g2V 04 by the replacement:
gV 03 → gV 03 − R3 ≡ gṼ 3
 ,
 g2V 04 → g2V 04 − R4 ≡ g2Ṽ4 .
 (316)
The 2PI effective action with the modified interaction is given by
δ W
 δ W
Γ Ṽ [φ , D] = W [J, R, R3, R4 ] −
 δ J
 J −
 δ R
 R .
 (317)
Since
δ ΓṼ
 δ W
 δ Γ Ṽ
 δ W
=
 ,
 =
 ,
 (318)
δ R3
 δ R3
 δ R4
 δ R4
one can express the remaining Legendre transforms, leading to Γ[φ , D,V 3 ,V 4], in terms
of the vertices Ṽ 3 , Ṽ 4 and V 03, V 04 :
δ Γ Ṽ [φ , D]
 δ Γ Ṽ
 [φ , D]
Γ[φ , D,V 3 ,V 4] = Γ Ṽ [ φ , D] −
 δ R3
 R3 − δ R4
 R4
δ Γ Ṽ
 [φ , D]
 δ Γ Ṽ
 [φ , D]
= Γ Ṽ [φ , D] − δ Ṽ 3
 ( Ṽ3 −V 03 ) − δ Ṽ 4
 ( Ṽ 4 −V 04 ) .
 (319)
What remains to be done is expressing Ṽ 3 and Ṽ4 in terms of V 3 and V 4. On the one hand,
from (307) and the definitions (308) and (309) one has
δ Γ Ṽ [φ , D]
 1
 3δ Γ gδ Ṽ [φ Ṽ 3
 , D]
 = − 6
 1 −ig D3V 3 + 3Dφ + φ 
 2 ,
 (320)
g2
 δ Ṽ 4
 = −
 24
 
 −ig2 2 D4V 4 4 − 3g2D5V3 − 4ig D3V 3φ + 3D2
+6Dφ + φ 
 .
 (321)
41
 Note that for φ 6= 0, in the phase with spontaneous symmetry breaking, φ ∼ O(1/g), and the three-loop
result (315) takes into account the contributions up to order g6 .
Introduction to Nonequilibrium Quantum Field Theory
 105
On the other hand, from the expansion of the 2PI effective action to three-loop order
with (315) one finds42
δ Γ Ṽ [φ , D]
 1
 3 1
 i
gδ Ṽ 3
 = − 6
 φ − 2
 Dφ + 6
 D3(gṼ 3 + g2 φ Ṽ 4)
1
 i
+ 4
 g2 D5 (gṼ 3 + g2φ Ṽ 4 ) Ṽ4 − 6
 D6 (gṼ 3 + g2φ Ṽ 4 )3
δ Γ Ṽ [φ , D]
 + O 1
 g 4 n (g2 1
 φ )m
 |n+m=5 2 1
 
 ,
 i
 (322)
g2
δ Ṽ 4
 = − 24
 φ − 4
 Dφ − 8
 D2 + 6
 D3 φ (gṼ 3 + g2 φ Ṽ 4 )
i 1
+ g2 D4
Ṽ 4 + g2 D5φ (gṼ 3 + g2 φ Ṽ 4) Ṽ 4
24
 4
1 i
+ 8
 D5
(gṼ 3 + g2
 φ Ṽ 4 )2 − 6
 D6 φ (gṼ 3 + g2 φ Ṽ 4 )3
Comparing (322) and (320) + yields
 O gn−2(g2 φ )m
 |n+m=6
 .
 (323)
3
gV 3 = (gṼ 3 + g2φ Ṽ 4 ) − 2 ig2 D2(gṼ 3 + g2 φ Ṽ 4) Ṽ 4 − D3(gṼ 3 + g2 φ Ṽ 4 )3
Similarly, for V 4 comparing + O gn (g2 (323) φ )m
 |n+m=5 and (321), 
 .
 and using (324) one finds
 (324)
This can be used to invert g2V4 the = above g2Ṽ 4 + relations O gn−2 as
 (g2 φ )m|n+m=6
 .
 (325)
gṼ 3 + g2 g φ 2
Ṽ Ṽ 4 4 = = gV g 2
V 34 + + 2
 3
 O ig3 g4 D2V .
 3V4 + g3D3V 3 3 + O g5 
 ,
 (327)
 (326)
Plugging this into (319) and expressing the free, 
 the one-loop and the Γ2 parts in terms
of V 3 and V 4 as well as V 03 and V 04 , one obtains from a straightforward calculation:
i
 i
Γ[φ , D,V 3 ,V 4] = S[φ ] + Trln D−1 + Tr D−1
 0 (φ )D + Γ2 [φ , D,V 3 ,V 4] ,
 (328)
2
 2
with
Γ2 [φ , D,V 3 ,V 4] = Γ02 [φ , D,V 3 ,V 4] + Γint
 2 [D,V 3,V 4] ,
 (329)
42 Note that since the exact φ -dependence of Γ2
 [φ , D] can be written as a function of (gV 03
 + g2 φ V 04
), the
parametrical dependence of the higher order terms in the variation of (315) with respect to (gV 03) is given
by O(gn (g2 φ )m|n+m=5 ) (cf. (322)).
Introduction to Nonequilibrium Quantum Field Theory
 106
1
 i
Γ02 [φ , D,V 3 ,V 4] = − 8
 g2D2V04 + 6
 gD3V 3 (gV 03 + g2 φ V 04 )
i 4 4
 1
 2+ g D V 4 V 04 + g4 D5V 3 V 04 ,
 (330)
24
 8
Γint
[D,V
 2
 3
 ,V
 4
]
 =
 −
 12
 i
 g
2 D
3V
 3
 2
 −
 48
 i
 g
4D
4V
 4
 2
 −
 24
 i g
4 D
6V
 3
 4
 +
 O
 g6 
 .
 (331)
The diagrammatic representation of these results is given in Figs. 19 and 21 of Sec. 6.2.1.
There the equivalent calculation is done for a SU (N) gauge theory and one has to replace
the propagator lines and vertices of the figures by the corresponding scalar propagator
and vertices. Note that for the scalar theory the thick circles represent the dressed
three-vertex gV 3 and four-vertex g2 V 4 , respectively, while the small circles denote the
corresponding effective classical three-vertex gV 03 + g2 φ V 04 and classical four-vertex
g2V 04 . As a consequence, the diagrams look the same in the absence of spontaneous
symmetry breaking, indicated by a vanishing field expectation value φ .
In (328), the action S[φ ] and D0 depend on the classical vertices as before. The
expression for Γ02 , which includes all terms of Γ2 that depend on the classical vertices,
is valid to all orders: Γint
 2 contains no explicit dependence on the field φ or the classical
vertices V 03 and V 04, independent of the approximation for the 4PI effective action. This
can be straightforwardly observed from (319), where the complete (linear) dependence
of Γ on V 03 and V 04 is explicit, together with (320) and (321).
6.1.2. Equivalence hierarchy for nPI effective actions
As pointed out in Sec. 1.1.2 of the introduction, for applications it is often desirable
to obtain a self-consistently complete description, which to a given order of a loop
or coupling expansion determines the nPI effective action Γ[φ , D,V 3,V4 , . . .,V n] for
arbitrarily high n. Despite the complexity of a general nPI effective action such a
description can be obtained in practice because of the equivalence hierarchy displayed
in Eq. (13): Typically the 2PI, 3PI or maybe the 4PI effective action captures already
the complete answer for the self-consistent description to the desired/computationally
feasible order of approximation. Higher effective actions, which are relevant beyond
four-loop order, may not be entirely irrelevant in the presence of sources describing
complicated initial conditions for nonequilibrium evolutions. However, their discussion
would be somewhat academic from the point of view of calculational feasibility and we
will concentrate on up to four-loop corrections or O(g6 ) in the following. Below we will
not explicitly write in addition to the loop-order the corresponding order of the coupling
g for the considered theory, which is detailed above in Sec. 6.1.1.
To show (13) we will first observe that to one-loop order all nPI effective actions
agree in the absence of sources. The one-loop result for the 1PI effective action is given
by (18) for vanishing source R. As has been explicitly shown in Sec. 2 (cf. Eq. (26)), the
one-loop 2PI effective action agrees with that expression, i.e.
Γ(1loop) [φ , D] = Γ(1loop) [φ ] ,
 (332)
Introduction to Nonequilibrium Quantum Field Theory
 107
in the absence of sources. The equivalence with the one-loop 3PI and 4PI effective
actions can be explicitly observed from the results of Sec. 6.1.1. In order to obtain
the 3PI expressions we could directly set the source R4 ≡ 0 from the beginning in the
computation of that section such that there is no dependence on V 4. Equivalently, we
can note from Eqs. (328)–(331) that already the 4PI effective action to this order simply
agrees with (18) for zero sources. As a consequence, it carries no dependence on V 3 and
V 4 , i.e.
Γ(1loop) [φ , D,V 3 ,V 4] = Γ(1loop) [φ , D,V 3 ] = Γ(1loop) [φ , D] .
 (333)
For the one-loop case it remains to be shown that in addition
Γ(1loop) [φ , D,V3 ,V 4 , . . .,V n ] = Γ(1loop)[φ , D,V 3,V4 ]
 (334)
for arbitrary n ≥ 5. For this we note that the number I of internal lines in a given loop
diagram is given by the number v3 of proper 3-vertices, the number v4 of proper 4-
vertices, . . . , the number vn of proper n-vertices in terms of the standard relation:
2I = 3v3 + 4v4 + 5v5 . . . + nvn ,
 (335)
where v3 + v5 + v7 + . . . has to be even. Similarly, the number L of loops in such a
diagram is
L==
I − v3 − v4 − v5 . . . − vn + 1
1
 v3 + v4 + 3
 v5 . . . +
 n − 2
 vn + 1 .
2
 2
 2
(336)
The equivalence (334) follows from the fact that for L = 1 equation (336) implies that
Γ(1loop) [φ , D,V 3,V 4, . . . ,V n] cannot depend in particular on V 5 , . . .V n .43
The two-loop equivalence of the 2PI and higher effective actions follows along the
same lines. According to (328)–(331) the 4PI effective action to two-loop order is given
by:
i
 i
Γ(2loop) [ φ , D,V 3 ,V 4 ] = S[φ ] + Trln D−1 + Tr D−1
 0 (φ )D
2
 2
+ Γ2
 (2loop)
 [φ , D,V 3 ,V 4 ] ,
 (337)
(2loop)
 1
 i
 i
 2Γ2
 [ φ , D,V 3 ,V 4 ] = − 8
 g2 D2V04 + 6
 gD3V 3 (gV 03 + g2 φ V 04 ) − 12
 g2D3V3 .
There is no dependence on V 4 to this order and, following the discussion above, there is
no dependence on V 5, . . . ,V n according to (336) for L = 2. Consequently,
Γ(2loop) [φ , D,V3 ,V 4 , . . . ,V n ] = Γ(2loop)[φ , D,V 3,V 4 ] = Γ(2loop) [φ , D,V 3] ,
 (338)
43
 Note that we consider here theories where there is no classical 5-vertex or higher, whose presence
would lead to a trivial dependence for the classical action and propagator.
Introduction to Nonequilibrium Quantum Field Theory
 108
for arbitrary n in the absence of sources. The latter yields
δ Γ(2loop) [ φ , D,V 3 ] δ Γ2
 (2loop)
 [φ , D,V 3 ]
 2δ V 3
 =
 δ V3
 = 0
 ⇒
 gV 3 = gV 03 + g φ V 04 ,
 (339)
which can be used in (337) to show in addition the equivalence of the 3PI and 2PI
effective actions (cf. Eq. (315)) to this order:
(2loop)
 1
 i
Γ2
 [φ , D,V 3 ] = − 8
 g2 D2V 04 + 12
 D3(gV 03 + g2 φ V 04)2
= Γ2
 (2loop)
 [φ , D] ,
 (340)
for vanishing sources. The inequivalence of the 2PI with the 1PI effective action to this
order,
Γ(2loop) [φ , D] 6= Γ(2loop) [φ ] ,
 (341)
follows from using the result of δ Γ2
 (2loop)
 [φ , D]/δ D = 0 for D in (340) in a straightfor-
ward way.44
In order to show the three-loop equivalence of the 3PI and higher effective actions,
we first note from (328)–(331) that the 4PI effective action to this order yields V 4 = V04
in the absence of sources:
δ Γ(3loop) [φ , D,V 3 ,V 4] δ Γ2
 (3loop)
 [φ , D,V 3 ,V 4]
 i
δ V 4
 =
 δ V 4
 = 24
 g4 D4 (V 04 −V 4 ) = 0 .
 (342)
Constructing the 3PI effective action to three-loop would mean to do the same calcu-
lation as in Sec. 6.1.1 but with V 4 → V 04 from the beginning (R4 ≡ 0). The result of a
classical four-vertex for the 4PI effective action to this order, therefore, directly implies:
Γ(3loop) [φ , D,V 3 ,V 4] = Γ(3loop) [φ , D,V 3 ] ,
 (343)
for vanishing sources. To see the equivalence with a 5PI effective action
Γ(3loop) [φ , D,V 3,V 4,V 5 ], we note that to three-loop order the only possible diagram
including a five-vertex requires v 3 = v5 = 1 for L = 3 in Eq. (336). As a consequence,
to this order the five-vertex corresponds to the classical one, which is identically zero
for the theories considered here, i.e. V 5 = V 05 ≡ 0. In order to obtain that (to this order
trivial) result along the lines of Sec. 6.1.1, one can formally include a classical five-
vertex V 05 and observe that the three-loop 2PI effective action admits a term ∼ D4V 05V3 .
After performing the additional Legendre transform the result then follows from setting
V 05 → 0 in the end. The equivalence with nPI effective actions for n ≥ 6 can again be
observed from the fact that for L = 3 Eq. (336) implies no dependence on V 6, . . .V n . In
addition to (343), we therefore have for arbitrary n ≥ 5:
Γ(3loop) [φ , D,V 3,V4 , . . . ,V n] = Γ(3loop) [φ , D,V 3,V 4] .
 (344)
44
 Here Γ(2loop)[φ , D] includes e.g. the summation of an infinite series of so-called “bubble” diagrams,
which form the basis of mean-field or Hartree-type approximations, and clearly go beyond a perturbative
two-loop approximation Γ(2loop) [φ ] (cf. Sec. 2.1).
Introduction to Nonequilibrium Quantum Field Theory
 109
The inequivalence of the three-loop 3PI and 2PI effective actions can be readily observed
from (328)–(331) and (343):
δ Γ(3loop) [φ , D,V 3]
 3 3 3δ V 3
 = 0
 ⇒
 gV 3 = g (V03 + g φ V 04 ) − g D V 3 .
 (345)
Written iteratively, the above self-consistent equation for V 3 sums an infinite number of
contributions in terms of the classical vertices. As a consequence, the three-loop 3PI
result can be written as an infinite series of diagrams for the corresponding 2PI effective
action, which clearly goes beyond Γ(3loop) [φ , D] (cf. Eq. (315)):
Γ(3loop) [φ , D,V 3] 6= Γ(3loop) [φ , D] .
 (346)
The importance of such an infinite summation will be discussed for the case of gauge
theories below.
6.2. Nonabelian gauge theory with fermions
We consider a SU (N) gauge theory with N f flavors of Dirac fermions with classical
action
Seff = S + S gf + SFPG
=
 Z
 d 4 x − 4
 1
 F μν
 a
 F μν a −
 2ξ
 1
 (G a (A))2 − ψ̄ (−iD
 / )ψ − η̄ a ∂μ (D μ η )a 
 , (347)
where ψ (ψ̄ ), A and η (η̄ ) denote the (anti-)fermions, gauge and (anti-)ghost fields,
respectively, with gauge-fixing term G a (A) = ∂ μ Aa μ for covariant gauges. The color
indices in the adjoint representation are a, b, . . . = 1, . . ., N 2 − 1, while those for the
fundamental representation will be denoted by i, j, . . . and run from 1 to N. Here
F μν
 a
 = ∂μ Aa ν − ∂ν Aa μ − g f abc A b μ Ac ν ,
 (348)
(Dμ η )a = ∂ μ η a − g f abc Aμ b η c ,
 (349)
D
 / = γ μ 
∂μ + igAa μ t a 
 ,
 (350)
where [t a,t b ] = i f abct c, tr(t at b ) = δ ab /2. For QCD, t a = λ a /2 with the Gell-Mann
matrices λ a (a = 1, . . ., 8). We will suppress Dirac and flavor indices in the following. It
is convenient to write (347) as
Seff
 =
 1
 2
 Z
xy
 A
μ a
 (x) iD0 −1 μν ab (x, y)Aν b(y) +
 Z
xy
 η̄ a (x) iG0 −1 ab (x, y)η b (y)
+
 Z
xy
 ψ̄i (x) i∆−1
 0 i j (x, y)ψ j (y) −
 6
 1
 g
 Z
xyz
 V 03
 abc
 μνγ (x, y, z)A μ a
(x)Aν b
 (y)Aγ c
 (z)
−
 24
 1 g
2
 Z
xyzw
 V 04
 abcd
 μνγδ (x, y, z, w)Aμ a
 (x)Aν b
 (y)Aγ c
 (z)Aδ d
 (w)
Introduction to Nonequilibrium Quantum Field Theory
 110
− g
 Z
xyz
 V 03 (gh)ab,c
 μ
 (x, y; z)η̄ a (x)η b (y)Aμ c(z)
− g
 Z
xyz
 V 03 (f)a
 μ i j (x, y; z)ψ̄i (x)ψ j (y)A μ a (z) ,
 (351)
with the free inverse fermion, ghost and gluon propagator in covariant gauges given by
i∆−1
 0 i j (x, y) = i ∂
 / x δi j δC (x − y) ,
 (352)
iG0 −1 ab (x, y) = −x δ ab δC (x − y) ,
 (353)
iD0 −1 μν ab (x, y) = 
g μν  − 1 − ξ −1 
 ∂μ ∂ν 
x δ abδC (x − y) ,
 (354)
where we have taken the fermions to be massless. The tree-level vertices read in coordi-
nate space:
V 03
 abc
 μνγ
 (x,
 y,
 z)
 =
 gμν f
 abc
 [δC 
 (y − z) ∂γ x δC (x − y) − δC (x − z) ∂γ yδC (y − x)]
+ gμγ [ δC (x − y) ∂ν z δC (z − x) − δC (y − z) ∂ν x δC (x − z)]
+ gνγ [δC (x − z) ∂μ y
 δC (y − x) − δC (x − y) ∂μ z δC (z − x)]
 ,
 (355)
V 04
 abcd
 μνγδ (x, y, z, w) =
 
 ace f abe bde
 f cde [gμγ gνδ − gμδ gνγ ]
 ade cbe
+ δC f
 (x f
 y)δC [gμν(x gγδ − z)δC gμδ (x gνγ ]w) + ,
 f
 f
 [g μγ gδ ν − g μν gγδ(356)
 ]

− − −V 03 (gh)ab,c
 μ
 (x, y; z) = − f abc ∂μ x δC (x − z)δC (y − z) ,
 (357)
V 03 (f)a
 μ i j (x, y; z) = γ μ ti a j δC (x − z)δC (z − y) .
 (358)
Note that V 03,abc μνγ
 (x, y, z) is symmetric under exchange of ( μ , a, x) (ν , b, y) (γ , c, z).
↔ ↔μνγδ
Likewise, V (x, y, z, w) is symmetric in its space-time arguments and under ex-
04,abcdchange of (μ , a) ↔ (ν , b) ↔ (γ , c) ↔ (δ , d).
In addition to the linear and bilinear source terms, which are required for a construc-
tion of the 2PI effective action, following Sec. 6.1 we add cubic and quartic source terms
to (351):
Ssource
 ′
 =
 1
 6 Z
xyz Rabc 3 μνγ
 (x, y, z)Aμ a (x)Aν b(y)Aγ c (z)
+
 24 1
 Z
xyzw Rabcd 4 μνγδ
 (x, y, z, w)Aμ a (x)Aν b(y)Aγ c (z)Aδ d (w)
+
 Z
xyz
 R3 (gh)ab,c
 μ
 (x, y; z)η̄ a (x)η b (y)Aμ c (z)
Introduction to Nonequilibrium Quantum Field Theory
 111
+
 Z
xyz
 R(f)a
 3 μ i j (x, y; z)ψ̄i (x)ψ j (y)A μ a (z) ,
 (359)
where the sources R3,4 obey the same symmetry properties as the corresponding classical
vertices V 03 and V 04 discussed above. The definition of the corresponding three- and four-
vertices follows Sec. 6.1. In particular, we have for the vertices involving Grassmann
fields:
δ R3 (gh)ab,c
 μ
 δ W
 (x, y; z)
 = −ig
 Z
x′ y′ z′
 D μ μ
 ′ cc′
 (z, z′ )Gba′
 (y, x′)V 3 (gh)a′b′ μ ′
 c′
 (x′ , y′ ; z′)Gb′
a(y′ , x),
δ R3 (f)a
 μδ i j W
 (x, y; z)
 = −ig
 Z
x′ y′ z′
 D μ μ
 ′ aa′
 (z, z′)∆ ji′ (y, x′ )V 3 (f)a′
 μ ′ i′ j′ (x′ , y′; z′ )∆ j ′ i (y′ , x) ,(360)
for the case of vanishing “background” fields hAi = hψ i = hψ̄ i = hη i = hη̄ i = 0, which
we will consider in the following.
6.2.1. Effective action up to four-loop or O(g6 ) corrections
Consider first the 2PI effective action with vanishing “background” fields, which
according to Sec. 2 can be written as
i
 i
Γ[D, ∆, G] =
 2
 Trln D−1 + 2
 Tr D−1
 0 D − iTr ln ∆
−1
 − iTr∆−1
 0 ∆
−iTr ln G−1 − iTrG−1
 0 G + Γ2 [D, ∆, G] .
 (361)
Here the trace Tr includes an integration over the time path C , as well as integration
over spatial coordinates and summation over flavor, color and Dirac indices. The exact
expression for Γ2 contains all 2PI diagrams with vertices described by (355)–(358) and
propagator lines associated to the full connected two-point functions D, G and ∆. In
order to clear up the presentation, we will give all diagrams including gauge and ghost
propagators only. The fermion diagrams can simply be obtained from the corresponding
ghost ones, since they have the same signs and prefactors.45 For the 2PI effective action
of the gluon-ghost system, Γ[D, G], to three-loop order the 2PI effective action is given
by (using the same compact notation as introduced in Sec. 6.1.1):
1
 i
 2
 i
 (gh) 2
 i
 2
Γ2 [D, G] = − 8
 g2 D2V 04 + 12
 g2 D3 V 03
 − 2
 g2 DG2V 03 + 48
 g4 D4V 04
1
 2
 i
 4
 i
 (gh) 3
+ 8
 g4 D5V 03
V04 − 24
 g4 D6V 03
 + 3
 g4D3 G3V 03 V 03
+ 4
 i
 g4 D2 G4V03 (gh) 4
 + O g6 
 .
 (362)
45
Note that to three-loop order there are no graphs with more than one closed ghost/fermion loop, such
that ghosts and fermions cannot appear in the same diagram simultaneously.
Introduction to Nonequilibrium Quantum Field Theory
 112
− 1
 8
 + 6 i
 + 24
 i
 + 18
FIGURE 19. The figure shows together with Fig. 20 the diagrammatic representation of
Γ02 [D, G,V 3,V 3 (gh)
 ,V 4 ] as given in Eq. (365). Here the wiggled lines denote the gauge field propagator
D and the unwiggled lines the ghost propagator G. The thick circles denote the dressed and the small ones
the classical vertices. This functional contains all terms of Γ2 that depend on the classical vertices gV 03,
gV03 (gh)
 and g2V 04 for an SU(N) gauge theory. There are no further contributions to Γ02 appearing at higher
order in the expansion. For the gauge theory with fermions there is in addition the same contribution as in
Fig. 20 with the unwiggled propagator lines representing the fermion propagator ∆ and the ghost vertices
(f)
 (f)
replaced by the corresponding fermion vertices V and V (cf. Eq. (358)).
03 3− i
FIGURE 20. Ghost/fermion part of Γ02 .
The result can be compared with (315) and taking into account an additional factor
of (−1) for each closed loop involving Grassmann fields (cf. Sec. 2.3). Here we have
suppressed in the notation the dependence of Γ2 [D, G] on the higher sources (359). The
desired effective action is obtained by performing the remaining Legendre transforms:
(gh)
 δ W
 δ W (gh) δ W
Γ[D, G,V 3,V3
 ,V 4] = Γ[D, G] −
 δ R3
 R3 − δ R
3
 (gh) R3 −
 δ R4
 R4 .
 (363)
The calculation follows the same steps as detailed in Sec. 6.1.1. For the effective action
to O(g6 ) we obtain:
(gh)
 i
 i
Γ[D, G,V 3 ,V 3
 ,V 4] =
 2
 Trln D−1 + 2
 Tr D−1
 0 D − iTr ln G
−1
 − iTrG−1
 0 G
(gh)
+Γ2 [D, G,V 3 ,V 3 ,V 4 ] ,
 (364)
with
Γ2[D, G,V 3 ,V 3
 (gh)
,V 4 ] = Γ02 [D, G,V 3,V 3
 (gh)
 ,V 4] + Γint
 2 [D, G,V 3 ,V 3
 (gh)
 ,V 4 ] ,
(gh)
 1
 i
 (gh) (gh)
Γ02[D, G,V 3 ,V 3
 ,V 4 ] = − 8
 g2D2V 04 + 6
 g2D3V3V 03 − ig2DG2V3 V 03
i 1
+ g4 D4
V 4V 04 + g4 D5V 3 2V 04 ,
 (365)
24
 8
(gh)
 i
 2 i
 (gh) 2
 i
 2
Γint
 2 [D, G,V 3 ,V 3
 ,V 4 ] = − 12
 g2 D3V 3 + 2
 g2DG2V 3
 − 48
 g4D4V 4i 4 6 4 i 4 3 3 (gh) 3
− 24
 g D V 3 + 3
 g D G V 3
 V 3
i
 (gh) 4
+ g4D2 G4 V3
 + O(g6) .
 (366)
4
Introduction to Nonequilibrium Quantum Field Theory
 113
i
 i
 i
− − − + O (g 6 )
12
 48
 24
FIGURE 21.
 The figure shows together with Fig. 22 the diagrammatic representation of
Γint
 2
 [D,
 G,V
 3,V
 3
 (gh)
 ,V
4 ] to three-loop order as given in Eq. (366). For the gauge theory with fermions,
to this order there is in addition the same contribution as in Fig. 22 with the unwiggled propagator lines
representing the fermion propagator ∆ and the ghost vertex replaced by the corresponding fermion vertex
(f)
V3 . This functional contains no explicit dependence on the classical vertices independent of the order of
approximation.
i
 i
 i
+ + + O (g 6 )
2
 3 4FIGURE 22. Ghost/fermion part of Γint
 2 to three-loop order.
The contributions are displayed diagrammatically in Figs. 19 and 20 for Γ02 , and in
Figs. 21 and 22 for Γint
 2 .
The equivalence of the 4PI effective action to three-loop order with the 3PI and nPI
effective actions for n ≥ 5 in the absence of sources follows along the lines of Sec. 6.1.2.
As a consequence, to three-loop order the nPI effective action does not depend on higher
vertices V 5 , V 6, . . . Vn . In particular with vanishing sources the four-vertex is given by the
classical one:
δ Γ(3loop) [D, G,V3,V 3
 (gh)
 ,V4 ]
 δ Γ2
 (3loop)
 [D, G,V3,V 3
 (gh)
,V4 ]
δ V 4
 =
 δ V 4
 = 0 ⇒ V4 = V 04 . (367)
If one plugs this into (365) and (366) one obtains the three-loop 3PI effective action,
Γ(3loop) [φ , D,V 3,V 3 (gh)
 ]. Similarly, to two-loop order one has
δ Γ(2loop) [D, G,V 3 ,V 3
 (gh)
]
 δ Γ2
 (2loop)
 [D, G,V3,V 3
 (gh)
]
δ V3
 =
 δ V 3
 = 0 ⇒ V 3 = V 03 ,
δ Γ(2loop) [D, G,V 3 ,V 3
 (gh)
]
 δ Γ2
 (2loop)
 [D, G,V3,V 3
 (gh)
]
 (gh)
 (gh)
δ V 3
 (gh)
 =
 δ V3
 (gh)
 = 0 ⇒ V 3
 = V 03 ,
(f)
and equivalently for the fermion vertex V3 . To this order, therefore, the combinatorial
factors of the two-loop diagrams of Fig. 19 and 21 for the gauge part, as well as of
Fig. 20 and 22 for the ghost/fermion part, combine to give the result (362) to two-loop
order for the 2PI effective action.
We have seen above that to two-loop order the proper vertices of the nPI effective
action correspond to the classical ones. Accordingly, at this order the only non-trivial
equations of motion in the absence of “background” fields are those for the two-point
Introduction to Nonequilibrium Quantum Field Theory
 114
Π (2) = − 2 i
 − 1
 2
 +
Σ (2) = −
FIGURE 23. The self-energy for the gauge field (Π) and the ghost/fermion (Σ) propagators as obtained
from the self-consistently complete two-loop approximation of the effective action. Note that at this order
all vertices correspond to the classical ones.
functions:
δΓ
 δΓ
 δΓ
= 0 ,
 = 0 ,
 = 0 ,
 (368)
δ D
 δ G
 δ∆
for vanishing sources. Applied to an nPI effective action (n > 1), as e.g. (364), one finds
for the gauge field propagator:
δ Γ2
D−1 = D−1
 0 − Π
 ,
 Π = 2i
 δ D
 .
 (369)
The ghost propagator and self-energy are
δ Γ2
G−1 = G−1
 0 − Σ
 ,
 Σ = −i
 δ G
 ,
 (370)
and equivalently for the fermion propagator ∆. (Cf. also Sec. 2 for the same relations
in the context of 2PI effective actions.) The self-energies to this order are shown in
diagrammatic form in Fig. 23. In contrast, for the three-loop effective action the three-
vertices get dressed and the stationarity conditions,
δΓ
 δΓ
 δΓ
= 0 ,
 = 0 ,
 = 0 ,
 (371)
δ V 3
 δ V 3
 (gh)
 δ V 3
 (f)
applied to (364)–(366) lead to the equations shown in the left graph of Fig. 25. Here
the diagrammatic form of the contributions is always the same for the ghost and for the
fermion propagators or vertices. We therefore only give the expressions for the gauge-
ghost system. If fermions are present, the respective diagrams have to be added in a
straightforward way.
The self-energies to this order are displayed in Fig. 24. It should be emphasized that
their relatively simple form is a consequence of the equations for the proper vertices,
Fig. 25. To see this we consider first the many terms generated by the functional
derivative of (365) and (366) with respect to the gauge field propagator:
δΓ
(3loop)
 2
 i
 1
Π
(3)
 ≡ 2i δ D
 = −
 2
 −
 +
 2
 + 2
1
 1
−
 −
 3
 +
 6
 + i
 (372)
i
 1
+
 4
 +
 2
 −2
 −
 .
Introduction to Nonequilibrium Quantum Field Theory
 115
Π (3) = − i
2+
+i
2−1
2
−1
6
Σ (3) = −
FIGURE 24. The self-energy for the gauge field (Π) and the ghost/fermion (Σ) propagators as obtained
from the self-consistently complete three-loop approximation of the effective action. (Cf. Fig. 25 for the
vertices.)
The short form for the self-energy of Fig. 24 is obtained through cancellations by
replacing in the above expression
1
 1
 1
 i
2
 =
 2
 −
 2
 −
 4
i
−
 2
 +
as well as
(373)
−
 = −
 +
 +
 .
 (374)
The latter equations follow from inserting the expressions for the dressed vertices of
Fig. 25. Noting in addition that the proper four-vertex to this order corresponds to
the classical one (cf. (367)) leads to the result. Along the very same lines a similar
cancellation yields the compact form of the ghost/fermion self-energy displayed in
Fig. 24.
6.2.2. Comparison with Schwinger-Dyson equations
The equations of motions of the last section are self-consistently complete to two-
loop/three-loop order of the nPI effective action for arbitrarily large n. We now compare
them with Schwinger-Dyson (SD) equations, which represent identities between n-point
functions. Clearly, without approximations the equations of motion obtained from an
exact nPI effective action and the exact (SD) equations have to agree since one can
always map identities onto each other. However, in general this is no longer the case for
a given order in the loop expansion of the nPI effective action.
By construction each diagram in a SD equation contains at least one classical vertex.
In general, this is not the case for equations obtained from the nPI effective action: The
Introduction to Nonequilibrium Quantum Field Theory
 116
Gauge three-vertex as well as ghost/fermionvertex from Γ(3loop) :
Compare: exact SD equation (ghost/fermion
diagrams not displayed):
1=
 −
 − i
2
1=
 −
 − i
2
−12
i
−12
i
1 1 1
− i
 − i
 +2
 2
 2
+
 +
1
 1
 i
+ + −2
 2
 2
=
 −
 −
i
 i
− −i
 +2
 6
FIGURE 25. Left: The gauge field three-vertex as well as the ghost (fermion) vertex as obtained from
the self-consistently complete three-loop approximation of the effective action. Apart from the isolated
classical three-vertex, all vertices in the equations correspond to dressed ones since at this order the four-
vertex equals the classical vertex. The result reflects the proper symmetry of the three-vertex. Right:
Schwinger-Dyson equation for the proper three-vertex V3 . Additional diagrams involving ghost or fermion
vertices are not displayed for brevity. We show it for comparison with the three-loop effective action result
displayed on the left. A naive truncation of the Schwinger-Dyson equation at the one-loop level does not
agree with the latter and symmetries (the second and third diagram contain a classical three-vertex instead
of a dressed one).
loop contributions of Γint
 in Eq. (366) or Figs. 21–22 are solely expressed in terms of
2full vertices. However, to a given loop-order cancellations can occur for those diagrams
in the equations of motion which do not contain a classical vertex. For the three-loop
effective action result this has been demonstrated above for the two-point functions.
Indeed, the equations for the two-point functions shown in Fig. 24 correspond to the SD
equations, if one takes into account that to the considered order the four-vertex is trivial
and given by the classical one (cf. Eq. (367)). However, such a correspondence is not
true for the proper three-vertex to that order.
As an example, we show on the right of Fig. 25 the SD equation for the proper
three-vertex, where for brevity we do not display the additional diagrams coming from
ghost/fermion degrees of freedom. One observes that a naive neglection of the two-loop
contributions of that equation would not lead to the effective action result for the three-
vertex shown in Fig. 25. Of course, the straightforward one-loop truncation of the SD
equation would not even respect the property of V 3 being completely symmetric in its
space-time and group labels. This is the well-known problem of loop-expansions of SD
equations, where one encounters the ambiguity of whether classical or dressed vertices
should be employed at a given truncation order.
We emphasize that these problems are absent using effective action techniques. The
fact that all equations of motion are obtained from the same approximation of the ef-
fective action puts stringent conditions on their form. More precisely, a self-consistently
Introduction to Nonequilibrium Quantum Field Theory
 117
complete approximation has the property that the order of differentiation of, say, Γ[D,V ]
with respect to the propagator D or the vertex V does not affect the equations of motion.
Consider for instance:
δ Γ[D,V = V (D)]
 δΓ
 δΓ
 δ V
=
 +
 .
 (375)
δ D
 δ D
 V
 δ V
 D δ D
If V = V (D) is the result of the stationary condition δ Γ/δ V = 0 then the above corre-
sponds to the correct stationarity condition for the propagator for fixed V : δ Γ/δ D = 0.
In contrast, with some ansatz V = f (D) that does not fulfill the stationarity condition
of the effective action, the equation of motion for the propagator would receive addi-
tional corrections ∼ δ V /δ D. In particular, it would be inconsistent to use the equation
of motion for the propagator δ Γ/δ D = 0 (cf. e.g. Fig. 24 which corresponds to the SD
equation result) but not the equation δ Γ/δ V = 0 for the vertex (cf. Fig. 25).
In turn, one can conclude that a wide class of employed truncations of exact SD equa-
tions cannot be obtained from the nPI effective action: this concerns those approxima-
tions which use the exact SD equation for the propagator but make an ansatz for the
vertices that differs from the one displayed in Fig. 25. The differences are, however, typ-
ically higher order in the perturbative coupling expansion and there may be many cases,
in particular in vacuum or thermal equilibrium, where some ansatz for the vertices is
a very efficient way to proceed. Out of equilibrium however, as mentioned above, the
“conserving” property of the effective action approximations can have important conse-
quences, since the effective loss of initial conditions and the presence of basic constants
of motion such as energy conservation is crucial.
6.2.3. Nonequilibrium evolution equations
Up to O(g6 ) corrections in the self-consistently complete expansion of the effective
action, the four-vertex parametrizing the diagrams of Figs. 24–25 corresponds to the
classical vertex. At this order of approximation there is, therefore, no distinction between
the coupling expansion of the 3PI and 4PI effective action. To discuss the relevant
differences between the 2PI and 3PI expansions for time evolution problems, we will
use the language of QED for simplicity, where no four-vertex appears. However, the
evolution equations of this section can be straightforwardly transcribed to the nonabelian
case by taking into account in addition to the equation for the gauge–fermion three-
vertex those for the gauge–ghost and gauge three-vertex (cf. Fig. 25). In the following
the effective action is a functional of the gauge field propagator Dμν (x, y), the fermion
propagator ∆(x, y) and the gauge-fermion vertex V 3 (f)
 μ (x, y; z), where we suppress Dirac
(f)
indices and we will write V 3 ≡ V . According to Eqs. (364)—(366) one has in this case
Γ2 [D, ∆,V ] = Γ02 [D, ∆,V ] + Γint
 2 [D, ∆,V ] ,
 (376)
with
Γ02
 = −ig
2
 Z
xyzu
 tr 
γ μ ∆(x, y)V ν (y, z; u)∆(z, x)D μν (x, u)
 ,
Introduction to Nonequilibrium Quantum Field Theory
(377)
118
where the trace acts in Dirac space. For the given order of approximation there are two
distinct contributions to Γint
 2 :
Γ2 (a)
 int
 = Γ2 i (a)
 2
 + Γ2 (b)
 + O g6 
 ,
 μν (378)
Γ2
 (b)
 =
 2
 i g
4
 Z
xyzuvw
 tr 
 Vμ (x, y; z)∆(y, u)V ν (u, v; w)∆(v, x)D (z, w)
 ,
Γ2
 =
 V 4
ρ g
 (x′ Z
, xyzuvwx′ y′ ; z′ )∆(y′ y′ z′ u′v′ , u′)V w′
 tr σ 
 (u′ Vμ (x, , v′; y; w′ z)∆(y, )∆(v′ , u)Vν x)D μρ (u, (z, v; z′)Dνσ w)∆(v, (w, x′)
 w′ ) .
The equations of motions for the propagators and vertex are obtained from the 
 station-
arity conditions (368) and (371) for the effective action. To convert (369) for the photon
propagator into an equation which is more suitable for initial value problems, we convo-
lute with D from the right and obtain for the considered case of vanishing “background”
fields, e.g. for covariant gauges (cf. also the discussion in Sec. 3.4):

g μ γ  − (1 − ξ −1 )∂ μ ∂γ
 
x
 Dγν (x, y) − i
 Z
 z
 Π = μig γ (x, μν z)D δC (x γν (z, y) y)
 .
 (379)
−Similarly, the corresponding equation of (370) yields the evolution equation for the
fermion propagator given in Eq. (141). Using the above results the self-energies are
Σ
( f )
 (x, y) = −g
2
 Z
z′ z′′
 Dμν (z′ , y)V μ (x, z′′ ; z′)∆(z′′, y)γ ν ,
 (380)
Π μν (x, y) = g2
 Z
z′ z′′
 tr γ μ ∆(x, z′ )V ν (z′, z′′; y)∆(z′′, x) .
 (381)
Note that the form of the self-energies is exact for known three-vertex. To see this
within the current framework, we note that the self-energies can be expressed in terms
of Γ02 only. The latter receives no further corrections at higher order in the expansion
(cf. Sec. 6.2.1), and thus the expression is exactly known: With
Z
 z
 Σ( f )
 (x, z)∆(z, y) = −i
 Zz 
 δ ∆(z, δ Γ02
 x)
 +
 δ δ ∆(z, Γint
 2
 x)
 
 ∆(z, y) ,
 (382)
and since Γint
 is only a functional of V ∆D
1/2 (cf. Sec. 6.2.1) one can use the identity
2Z
z δ δ ∆(z, Γint
 2
 x)
 ∆(z, y) =
 Z
zz′
 Vμ (x, z; z′ )
 V μδ (y, Γint
 2
 z; z′)
= −
 Z
zz′
 Vμ (x, z; z′ )
 V μ (y, δ Γ02
 z; z′)
 (383)
Introduction to Nonequilibrium Quantum Field Theory
 119
to express everything in terms of the known46 Γ02 . The last equality in (383) uses that
δ (Γ02 + Γint
 2 )/δ ∆ = 0. A similar discussion can be done for the photon self-energy. As
a consequence, all approximations are encoded in the equation for the vertex, which is
obtained from (378) as
V μ (x, y; z) = V 0 μ
 (x, y; z) − g2
 Z
vwx′y′ u′w′
 V ν (x, v; w)∆(v, x′ )V μ (x′, y ′ ; z)
where
 ∆(y′, u′ )V σ (u′, y; w′ )Dσ ν (w′, w) + O g4
 ,
 (384)
V 0 μ
 (x, y; z) = γ μ δ (x z)δ (z y) .
 (385)
− −For the self-consistently complete two-loop approximation the self-energies are given
by
Πμν Σ( f ) (x, (x, y) y) = = g2 −g2Dμν tr γ μ ∆(x, (x, y)γ y)γ ν μ ∆(y, ∆(x,x) y)γ + ν O + O g4 g4 .
 
 ,
 (386)
 (387)
Following the discussion of Sec. 3.4.1, we decompose the two-point 
 functions into spec-
tral and statistical components using the identities (143) for gauge fields and (121) for
fermions. Then ρD corresponds to the gauge field spectral function and F D is the statis-
tical two-point function, while ρ ( f ) and F ( f ) are the corresponding fermion two-point
functions. The same decomposition can be done for the corresponding self-energies:
μν μν
 i μν
Π (x, y) = Π(F) (x, y) − 2
 Π(ρ ) (x, y) sign(x0 − y0 ) ,
 (388)
f ( f )
 i
 ( f )
Σ( )(x, y) = ΣF (x, y) − 2
 Σρ (x, y) sign(x0 − y0 ) ,
 (389)
and similarly for the fermions as described by (126). Since the above decomposition for
the propagators and self-energies makes the time-ordering explicit, we can evaluate the
r.h.s. of (379) along the time contour following the discussion of Sec. 3.4.3. One finds
the evolution equations:
 gμ
 γ  − (1 − ξ −1)∂ μ ∂γ x ρD γν
 (x, y) =
 Z
y0
 x0
 dz Π(ρ μγ
 ) (x, z)ρD,γ ν (z, y) ,
 (390)

gμ
 γ  − (1 − ξ
 −1
)∂ μ ∂γ
 
x F D
 γν
 (x, y)
 =
 Z t0
 x0
 y0
 dz Π(ρ μγ
 ) (x, z)F D,γ ν (z, y)
−
 Z t0
 dz Π(F)(x, μγ
 z)ρD,γ ν (z, y) ,
 (391)
denoted where we by use t0 , the which abbreviated was taken notation without R
t loss t 1 2 dz ≡ of R
 generality t t 1 2 dz0 R
−∞
 ∞ d to d
 z. be Here zerothe in the initial respective
 time is
46
 This can also be directly verified from (377) to the given order of approximation.
Introduction to Nonequilibrium Quantum Field Theory
120
equations (138) for scalars. The equations of motion for the fermion spectral and statis-
tical correlators are obtained in a similar way from (141) as described in Sec. 3.4.3 and
are given in Eq. (142).
A similar discussion as for the two-point functions can also be done for the higher
correlation functions. For the three-vertex we write
μ μ
 μV (x, y; z) = V0 (x, y; z) + V̄ (x, y; z) .
 (392)
and the corresponding decomposition into spectral and statistical components reads
μV̄ (x, y; z) =
μ
 i μ
U(F)(x, y; z) sign(y0 − x0 ) sign(z0 − x0 ) − 2
 U(ρ )(x, y; z) sign(y0 − z0 )
μ
 0
 0
 0
 0
 i μ
+ V(F) (x, y; z) sign(x − z ) sign(y − z ) − 2
 V (ρ ) (x, y; z) sign(x0 − y0 )
μ
 0
 0
 0
 0
 i μ
+ W (F) (x, y; z) sign(z − y ) sign(x − y ) − 2
 W(ρ ) (x, y; z) sign(z0 − x0 ). (393)
To discuss this in more detail we use the short-hand notation
Θ(x0 , y0 , z0) ≡ Θ(x0 − y0 )Θ(y0 − z0 ) .
 (394)
With the separation of Eq. (392), the time-ordered three-vertex can be written as (cf. also
the corresponding discussion for two-point functions in Sec. 3.4.1)
μ μ
 μ
V̄ (x, y; z) = V (x, y; z)Θ(x0 , y0 , z0) +V (x, y; z)Θ(y0 , z0, x0 )
(1) (2)μ
 0 0 μ
 0+ V (x, y; z)Θ(z , x , y0) +V (x, y; z)Θ(z0, y0, x )
(3) (4)μ
 μ
+ V (x, y; z)Θ(x0 , z0, y0) +V (x, y; z)Θ(y0 , x0, z0 ) ,
 (395)
(5) (6)μ
with ‘coefficients’ V (x, y; z), i = 1, . . . , 6. These coefficients can be expressed in terms
(i)μ
 μ
 μ
of three spectral vertex functions U (x, y; z), V (x, y; z) and W (x, y; z), as well as the
(ρ ) (ρ ) (ρ )μ
 μ
 μ
corresponding statistical components U (x, y; z), V (x, y; z) and W (x, y; z) that have
(F) (F) (F)been employed in Eq. (393). One finds, suppressing the space-time arguments:
V (1) μ
 μ
 ≡ U(F) μ
 μ
 +V (F) μ
 μ
 −W (F) μ
 μ
 − 2
 i i  U(ρ μ
 μ
 ) +V (ρ μ
 μ
 ) −W (ρ μ
 μ
 ) 
 ,
V (2) μ
 ≡ U(F) μ
 −V (F) μ
 +W (F) μ
 − 2
  iU(ρ ) −V μ
 (ρ ) +W μ
 (ρ ) 
 ,
 μ
V (3) μ
 ≡ −U(F) μ
 +V μ
 (F) +W μ
 (F) −
 i
 2
 
μ
 −U(ρ ) μ
 +V (ρ ) μ
 +W (ρ ) 
 ,
V (4) μ
 ≡ U(F) μ
 +V (F) μ
 −W (F) μ
 + 2
 i 
 U(ρ μ
 ) +V (ρ μ
 ) −W (ρ μ
 ) 
 ,
 (396)
V (5) μ
 ≡ U(F) μ
 −V (F) μ
 +W (F) μ
 + 2
  iU(ρ ) −V μ
 (ρ ) +W μ
 (ρ ) 
 ,
 μ
V (6) ≡ −U(F) +V (F) +W (F) +
 2
 
 −U(ρ ) +V (ρ ) +W (ρ ) 
 .
Introduction to Nonequilibrium Quantum Field Theory
 121
μ
In terms of the coefficients V these are given by:
(i)U (F) μ
 μ
 =
 1 1 4
  V (1) μ
 μ
 +V (2) μ
 μ
 +V(4) μ
 μ
 +V (5) μ
 μ
 
 , U (ρ μ
 μ
 ) = i 2
 i  V(1) μ
 μ
 +V (2) μ
 μ
 −V (4) μ
 μ
 −V (5) μ
 μ
 
 ,
V (F) μ
 =
 4
 1
  V (1) μ
 +V (3) μ
 +V(4) μ
 +V (6) μ
 
 , V (ρ μ
 ) = 2
 i  V (1) μ
 +V (3) μ
 −V (4) μ
 −V(6) μ
 
 ,
W (F) =
 4
 
 V (2) +V (3) +V(5) +V (6)
 , W (ρ ) = 2
 V (2) +V (3) −V (5) −V (6)
 .
Insertion shows the equivalence of (395) and (393).
6.3. Kinetic theory
To make contact with frequent discussions in the literature, we will consider for the
above gauge field and fermion nonequilibrium equations a standard “on-shell” approx-
imation which is typically employed to derive kinetic equations for effective particle
number densities. This part can be viewed as a continuation of Sec. 4.1.3, where scalar
fields and the limitations of “on-shell” approximations have been discussed.
6.3.1. “On-shell” approximations
The evolution equations (390)–(142) to order g2 and higher contain “off-shell” and
“memory” effects due to their time integrals on the r.h.s. (cf. also Sec. 4.1). To simplify
the description one conventionally considers a number of additional assumptions which
finally lead to effective kinetic or Boltzmann-type descriptions for “on-shell” particle
number distributions. The derivation of kinetic equations for the two-point functions
F μν (x, y) and ρ μν (x, y) of Sec. 6.2.3 can be based on (i) the restriction that the initial
condition for the time evolution problem is specified in the remote past, i.e. t0 → −∞,
(ii) a derivative expansion in the center variable X = (x + y)/2, and (iii) a “quasiparticle”
picture. In contrast to the discussion for scalar fields in Sec. 4.1.3, within this approach
one first sends the initial time t 0 to the remote past in the equations (390)–(142). This
allows one to use standard derivative expansion techniques in a straightforward way. The
procedure of Sec. 4.1.3 has the advantage that one can discuss which contributions to
the evolution are lost in this limit. On the other hand, the advantage of the derivative ex-
pansion is that it may in principle be used to include higher order corrections. However,
the complexity of a derivative expansion grows rapidly beyond the lowest order.
For the sake of simplicity (not required), we consider the Feynman gauge ξ = 1
in the following. We will also consider a chirally symmetric theory, i.e. no vacuum
fermion mass, along with parity and CP invariance. Therefore, the system is charge
neutral and, in particular, the most general fermion two-point functions can be written in
terms of vector components only: F ( f ) (x, y) = γμ F ( f )μ (x, y), ρ ( f ) (x, y) = γμ ρ ( f ) μ (x, y),
with hermiticity properties F ( f )μ (x, y) = [F ( f )μ (y, x)]∗, ρ ( f )μ (x, y) = −[ ρ ( f )μ (y, x)]∗ . For
the gauge fields the respective properties of the statistical and spectral correlators read
F D μν
 (x, y) = [F D νμ
 (y, x)]∗ , ρD μν
 (x, y) = −[ρD νμ
 (y, x)]∗ .
Introduction to Nonequilibrium Quantum Field Theory
 122
In order to Fourier transform with respect to the relative coordinate sμ = xμ − yμ , we
write
F̃ D μν
 μν
 (X , k) =
 Z
 d4 s eiks F D μν
 
 μν
 X + 2
 s
 , X s
 −
 2
 s 
 s,
 (397)
ρ̃D (X , k) = −i Z
 d4 s eiks ρD 
X + 2
 , X −
 2
 
 ,
 (398)
and equivalently for the fermion statistical and spectral function, F̃ ( f ) (X , k) and
ρ̃ ( f ) (X , k). Here we have introduced a factor −i in the definition of the spec-
tral function transform for convenience. For the Fourier transformed quantities
μν
we note the following hermiticity properties, for the gauge fields: [F̃ D (X , k)]∗ =
F̃ D νμ
 (X , k) , [ρ̃D μν
 (X , k)] ∗ = ρ̃D νμ
 (X , k), and for the vector components of the fermion
fields: [F̃ ( f ) μ (X , k)]∗ = F̃ ( f )μ (X , k) , [ρ̃ ( f )μ (X , k)]∗ = ρ̃ ( f ) μ (X , k). After sending
t0 → −∞ the derivative expansion can be efficiently applied to the exact Eqs. (390)—
(142). Here one considers the difference of (390) and the one with interchanged
coordinates x and y, and equivalently for the other equations. We use
Z
 d4s eiks
 Z
 d4z f (x, z)g(z, y) = f  ̃ (X , k)g̃(X , k) + . . .
 (399)
Z
 d4 s e iks
 Z
 d4 z
 Z
 d4 z′ f (x, z)g(z, z′)h(z′ , y) = f  ̃ (X , k)g̃(X , k)h̃(X , k) + . . .
where the dots indicate derivative terms, which will be neglected. E.g. the first derivative
corrections to (399) can be written as a Poisson bracket, which is in particular important
if “finite-width” effects of the spectral function are taken into account. However, a
typical quasiparticle picture which employs a free-field or “zero-width” form of the
spectral function is consistent with neglecting derivative terms in the scattering part.
We also note that the quasiparticle/free-field form of the two-point functions implies
F D μν
 (X , k) → −g μν F D (X , k) ,
 ρD μν
 (X , k) → −gμν ρD(X , k) .
 (400)
At this point the only use of the above replacement is that all Lorentz contractions can be
done. This doesn’t affect the derivative expansion but keeps the notation simple. Similar
to Eq. (398), we define the Lorentz contracted self-energies:
− 4Π̃(F)(X , k) ≡
 Z
 d4 s eiks 4 Π(F)μ iks μ
 μ
 
X + 2
 s
 , X s
 −
 2
 s 
s ,
 (401)
−4Π̃(ρ )(X , k) ≡ −i Z
 d s e Π(ρ )μ 
X + 2
 , X −
 2
 
 .
 (402)
Without further assumptions, i.e. using the above notation and applying the approxima-
tion (399) and (400) to the exact evolution equations one has47
μ
 ∂
2 k ∂ X μ
 F̃ D(X , k) = Π̃(ρ )(X , k) F̃ D (X , k) − Π̃(F) (X , k) ρ̃D(X , k) ,
 (403)
47
 The relation to a more conventional form of the equations can be seen by writing:
Π̃(ρ )F̃ D − Π̃(F) ρ̃D
 (X, k) =
Introduction to Nonequilibrium Quantum Field Theory
123
μ
 ∂
2 k ∂ X μ
 ρ̃D(X , k) = 0 .
 (404)
One observes that the equations (403) and (404) have a structure reminiscent of that
for the exact equations for vanishing “background” fields, (390) and (391), evaluated
at equal times x0 = y0 . However, one should keep in mind that (403) and (404) are, in
particular, only valid for initial conditions specified in the remote past and neglecting
gradients in the collision part.
From (404) one observes that in this approximation the spectral function receives
no contribution from scattering described by the r.h.s. of the exact equation (390). As a
consequence, the spectral function obeys the free-field equations of motion. In particular,
ρD μν
 (x, y) have to fulfill the equal-time commutation relations [ρD μν
 (x, y)]x0 =y0 = 0 and
[∂x0 ρD μν
 (x, y)]x0=y0 = −gμν δ (x − y) in Feynman gauge. The Wigner transformed free-
field solution solving (404) then reads ρ̃D (X , k) = ρ̃D (k) = 2π sign(k0 ) δ (k2). A very
similar discussion can be done as well for the evolution equations (142) for fermions,
which is massless due to chiral symmetry as stated above. Again, in lowest order in
the derivative expansion the fermion spectral function obeys the free-field equations of
motion and one has ρ̃ ( f )(X , k) = ρ̃ ( f ) (k) = 2π / k sign(k0) δ (k2 ).
Assuming a “generalized fluctuation-dissipation relation” or so-called “Kadanoff-
Baym ansatz”:
F̃ D (X , k) =
 
 2
 1
 + nD (X , k)
 ρ̃D (X , k) ,
F̃ ( f )
 (X , k) =
 
 1
 2
 − n( f )
 (X , k)
 ρ̃ ( f )(X , k) ,
 (405)
one may extract the kinetic equations for the effective photon and fermion particle num-
bers nD and n( f ) , respectively. Considering spatially homogeneous, isotropic systems for
simplicity, we define the on-shell quasiparticle numbers (t ≡ X 0 )
nD (t, k) ≡ nD (t, k)|k0=k
 ,
 n( f )(t, k) ≡ n( f ) (t, k)|k0=k
 (406)
and look for the evolution equation for nD (t, k). Here it is useful to note the symmetry
properties
F̃ D(t, −k) = F̃ D (t, k) , ρ̃D (t, −k) = −ρ̃D (t, k) ,
F̃ ( f ) (t, −k)μ = −F̃ ( f ) (t, k)μ , ρ̃ ( f ) (t, −k)μ = ρ̃ ( f ) (t, k) μ .
 (407)
Applied to the quasiparticle ansatz (405) these imply
nD (t, −k) = − [nD(t, k) + 1] , n( f )(t, −k) = − h
n( f ) (t, k) − 1i
 .
 (408)

Π̃(F) + 2
 1
 Π̃(ρ )
 
F̃ D − 2
 1
 ρ̃D  − 
Π̃ (F) − 2
 1
 Π̃(ρ ) 
F̃ D + 2
 1
 ρ̃D
 (X, k) .
The difference of the two terms on the r.h.s. can be directly interpreted as the difference of a so-called
“loss” and a “gain” term in a Boltzmann-type description.
Introduction to Nonequilibrium Quantum Field Theory
 124
FIGUREvertices.
26.Infinite series of self-energy contributions with dressed propagator lines and classical
This is employed to rewrite terms with negative values of k0. To order g2 the self-
energies read (cf. Eq. (387)):
Π̃(F) (X , k) = 2g
2
 Z
 (2π d4 p )4
 hF̃ ( f )
 (X , k + p) μ F̃ ( f )(X , p) μ
− 1
 4
 ρ̃ ( f )(X , k + p) μ ρ̃ ( f ) (X , p)μ i
 ,
Π̃(ρ ) (X , k) = 2g
2
 Z
 (2π d4 p )4
 hF̃ ( f )
 (X , k + p) μ ρ̃ ( f )(X , p) μ
−ρ̃ ( f )(X , k + p) μ F̃ ( f )(X , p)μ i
 .
 (409)
From the equations (403) and (405) one finds at this order: (q ≡ k − p)
∂t nD (t, k) = g2 k 2
 Z
 (2π d
3 p
 )3 2k2p2q
 1
 (

n( f ( )(t, f )
 p) n( f )(t, q) [nD(t, ( f )
 k) + 1]
− h
n (t, p) − 1ih
n (t, q) − 1i
 nD (t, k)
2πδ (k − p − q)
+ 2h
f n( f )
 (t, p) − f ) 1i
 n( f )(t, q) [nD(t, k) + 1]
−n( )(t, p) h
n( (t, q) − 1i
 nD(t, k)
 2πδ (k + p − q)
+
 h
n( f )(t, p) − 1ih
n( f ) (t, q) − 1i
 [nD(t, k) + 1]
−n( f )(t, p) n( f ) (t, q) nD (t, k)
2πδ (k + p + q))
 .
 (410)
The r.h.s. shows the standard “gain term” minus “loss term” structure. E.g. for the case
k2 > 0, k0 > 0 the interpretation is given by the elementary processes eē → γ , e → eγ ,
ē → ēγ and “0” → eēγ from which only the first one is not kinematically forbidden.
From (410) one also recovers the fact that the “on-shell” evolution with k2 = 0 vanishes
identically at this order. A nonvanishing result is obtained if one takes into account
“off-shell” corrections for a fermion line in the loop of the self-energy (409). As a
consequence the first nonzero contribution to the self-energy starts at O(g4).
Since the lowest order contribution to the kinetic equation is of O(g4 ), the 3PI
effective action provides a self-consistently complete starting point for its description.
Introduction to Nonequilibrium Quantum Field Theory
 125
At this order the self-energies and vertex are given by Eqs. (380), (381) and (384).
Starting from the three-vertex (384) consider for a moment the vertex resummation for
the photon leg only, i.e. approximate the fermion-photon vertex by the classical vertex.
As a consequence, one obtains:
V μ (x, y; z) ≃ γ μ δ (x − z)δ (z − y)
 (411)
−g2
 Z
x′y′
 γ ν ∆(x, z)V μ (x′, y ′ ; z)∆(y′ , y)γ σ Dσ ν (y, x) .
Using this expression for the photon self-energy (381), by iteration one observes that
this resums all the ladder diagrams shown in Fig. 26. Here propagator lines correspond
to self-energy resummed propagators whereas all vertices are given by the classical
ones. In the context of kinetic equations, relevant for sufficiently homogeneous systems,
the dominance of this sub-class of diagrams has been discussed in detail in the weak
coupling limit in the literature (cf. the bibliography at the end of this section). One
may decompose the contributions to the kinetic equation into 2 ↔ 2 particle processes,
such as eē → γγ annihilation in the context of QED, and inelastic “1 ↔ 2” processes,
such as the nearly collinear bremsstrahlung process. For the description of “1 ↔ 2”
processes, once Fourier transformed with respect to the relative coordinates, the gauge
field propagator in (411) is required for space-like momenta. Furthermore, as seen from
(410), the proper inclusion of nonzero contributions from 2 ↔ 2 processes requires to
go beyond the naive on-shell limit. In the context of the evolution equations (390) and
(391) this can be achieved by employing the following identities:
F D μν
 (x, y)
 =
 t0→−∞ lim
 ∞
 Z t0
 x0
 dz
 Zt0
 y0
 dz′ 
ρD (x, z)Π(F)(z, z′)ρD (z′, y)
 μν
= −
 Z−∞
 dzdz′ x0
 
DR y0
 (x, z)Π(F) (z, z′ )DA (z′ , y)
 μν
ρD μν
 (x, y)
 =
 t0→−∞ lim
 ∞
 Z t0
 dz
 Zt0
 dz′ 
ρD (x, z)Π(ρ )(z, z′)ρD(z′ , y)
μν
= −
 Z−∞
 dzdz′ 
DR (x, z)Π(ρ )(z, z′ )DA(z′ , y)
μν
 ,
 (412)
y0 written )ρD (x, in y)μν terms and DA of (x, the y)μν retarded = and advanced x0 )ρD (x, y)μν propagators, , in order DR to have (x, y)μν an unbounded
 = Θ(x0 −
−Θ(y0 −time integration. The above identity follows from a straightforward application of the
exact evolution equations and using the anti-symmetry property of the photon spectral
function, ρD μν
 (x, y)|x0=y0 = 0. We emphasize that the identity does not hold for an initial
value problem where the initial time t0 is finite. Similarly, one finds from (142) for the
fermion two-point functions using γ 0ρ ( f ) (x, y)|x0 =y0 = iδ (x − y):
F ( f )(x, y) = −
 Z−∞
 ∞
 ∞
 dzdz′ ∆R (x, z)Σ(F) ( f )
 (z, z′)∆A (z′, y) ,
ρ
 ( f )
(x, y) = −
 Z−∞
 dzdz′ ∆R (x, z)Σ( (ρ f ) )
 (z, z′)∆A (z′, y) ,
 (413)
Introduction to Nonequilibrium Quantum Field Theory
 126
with ∆R (x, y) = Θ(x0 − y0 )ρ ( f ) (x, y) and ∆A(x, y) = −Θ(y0 − x0 )ρ ( f ) (x, y). Neglecting
all derivative terms, i.e. using (399), and the above notation these give:48
F̃ D (X , k) ≃ D̃R (X , k)Π̃(F) (X , k)D̃ A (X , k) ,
ρ̃D (X , k) ≃ D̃R (X , k)Π̃(ρ )(X , k)D̃A (X , k) ,
 (414)
and equivalently for the fermion two-point functions. Applied to one fermion line in the
one-loop contribution of Fig. 26, it is straightforward to recover the standard Boltzmann
equation for 2 ↔ 2 processes, using the O(g2) fermion self-energies:
Σ̃(F) (X , k)μ = −2g2
 Z
 (2π d4 p )4
 h
F̃ D (X , p)F̃ ( f )(X , k − p)μ
+ 1
 4
 ρ̃D (X , p)ρ̃ ( f )(X , k − p)μ i
 ,
Σ̃(ρ ) (X , k)μ
 = −2g
2
 Z
 (2π d4 p )
4
 h
F̃ D (X , p)ρ̃ ( f )(X , k − p)μ
+ ρ̃D (X , p)F̃ ( f )
(X , k − p) μ
 i
 .
 (415)
For the Boltzmann equation ∆R and ∆A are taken to enter the scattering matrix element,
which is evaluated in (e.g. “hard thermal loops” resummed) equilibrium, whereas all
be other obtained lines are with taken the help to be of “on-shell”. (414) with The the O(g2 contributions ) photonfrom self-energies the 1 ↔ (409). 2 processes Of course,
 may
simply adding the contributions from 2 ↔ 2 processes and 1 ↔ 2 processes entails the
problem of double counting since a diagram enters twice. This occurs whenever the
internal line in a 2 ↔ 2 process is kinematically allowed to go on-shell and has to be
suppressed.
6.3.2. Discussion
In view of the generalized fluctuation-dissipation relation (405) employed in the above
“derivation”, one could be tempted to say that for consistency an equivalent relation
should be valid for the self-energies as well:
Π̃ (F) (X , k) =
 
 2
 1
 + nD (X , k)
 Π̃ (ρ )(X , k) .
 (416)
Such a relation is indeed valid in thermal equilibrium, where all dependence on the
center coordinate X is lost. Furthermore, the above relation can be shown to be a
consequence of (405) using the identities (412) in a lowest-order derivative expansion:
Together with Eq. (414) the above relation for the self-energies is a direct consequence
48
 As for the spectral function ρ (X, k) in Eq. (398), the Fourier transform of the retarded and advanced
propagators includes a factor of −i.
Introduction to Nonequilibrium Quantum Field Theory
 127
of the ansatz (405). However, clearly this is too strong a constraint since the evolution
equation (403) would become trivial in this case: Eq. (405) and (416) lead to a vanishing
r.h.s. of the evolution equation for F̃ D(X , k) and there would be no evolution.
The above argument is just a manifestation of the well-known fact that the kinetic
equation is not a self-consistent approximation to the dynamics. The discussion of
Sec. 6.3.1 takes into account the effect of scattering for the dynamics of effective
occupation numbers, while keeping the spectrum free-field theory like. In contrast, the
same scattering does induce a finite width for the spectral function in the self-consistent
nPI approximation discussed in Sec. 6.2.3 because of a nonvanishing imaginary part of
the self-energy (cf. also the discussion and explicit solution of a similar Yukawa model
in Sec. 4.2 and the discussion in Sec. 4.1).
Though particle number is not well-defined in an interacting relativistic quantum
field theory in the absence of conserved charges, the concept of time-evolving effective
particle numbers in an interacting theory is useful in the presence of a clear separation
of scales. Much progress has been achieved in the quantitative understanding of kinetic
descriptions in the vicinity of thermal equilibrium for gauge theories at high temperature,
which is well documented in the literature and for further reading we refer to the
bibliography below.
For gauge theories the employed on-shell limit circumvents problems of gauge in-
variance or subtle aspects of renormalization. We emphasize that renormalization for
linear symmetries as realized in QED can be treated along similar lines as discussed in
Sec. 2.2. The generalization to nonabelian gauge theories is, however, technically more
involved and needs to be further investigated.
A derivative expansion is typically not valid at early times where the time evolution
can exhibit a strong dependence on X , and the homogeneity requirement underlying
kinetic descriptions may only be fulfilled at sufficiently late times. This has been exten-
sively discussed in the context of scalar and fermionic theories in Sec. 4. Homogeneity
is certainly realized at late times sufficiently close to the thermal limit, since for thermal
equilibrium the correlators do strictly not depend on X . Of course, by construction ki-
netic equations are not meant to discuss the detailed early-time behavior since the initial
time t0 is sent to the remote past. For practical purposes, in this context one typically
specifies the initial condition for the effective particle number distribution at some fi-
nite time and approximates the evolution by the equations with t0 → −∞. The role of
finite-time effects has been controversially discussed in the recent literature in the con-
text of photon production in relativistic plasmas at high temperature. Here a solution of
the proper initial-time equations as discussed in Sec. 6.2.3 seems mandatory.
7. ACKNOWLEDGEMENTS
I thank Gert Aarts, Daria Ahrensmeier, Rudolf Baier, Szabolcs Borsányi, Jürgen Cox,
Markus M. Müller, Urko Reinosa, Julien Serreau and Christof Wetterich for very fruitful
collaborations on this topic.
Introduction to Nonequilibrium Quantum Field Theory
 128
8. BIBLIOGRAPHICAL NOTES
I apologize for the omission of many interesting contributions to this wide topic in the
annotated list below, which concentrates on relativistic quantum field theory applications
related to the content of the text presented above.
•
 A rather recent short review with a more comprehensive list of references can be
found in: Progress in nonequilibrium quantum field theory, J. Berges and J. Serreau,
in Strong and Electroweak Matter 2002, ed. M.G. Schmidt (World Scientific, 2003),
http://arXiv:hep-ph/0302210.
•
 General discussions on nPI effective actions include: Effective Action For Com-
posite Operators, J. M. Cornwall, R. Jackiw and E. Tomboulis, Phys. Rev. D 10
(1974) 2428. Higher effective actions for bose systems, H. Kleinert, Fortschritte
der Physik 30 (1982) 187. Functional Methods in Quantum Field Theory and
Statistical Physics, A.N. Vasiliev, Gordon and Breach Science Pub. (1998). nPI
effective action techniques for gauge theories, J. Berges, Phys. Rev. D in print,
http://arXiv:hep-ph/0401172. Gauge-fixing dependence of Phi-derivable approxi-
mations, A. Arrizabalaga and J. Smit, Phys. Rev. D 66 (2002) 065014.
•
 The nonperturbative 2PI 1/N expansion is derived in: Controlled nonperturbative
dynamics of quantum fields out of equilibrium, J. Berges, Nucl. Phys. A699 (2002)
847; Far-from-equilibrium dynamics with broken symmetries from the 1/N expan-
sion of the 2PI effective action, G. Aarts, D. Ahrensmeier, R. Baier, J. Berges and
J. Serreau, Phys. Rev. D66 (2002) 045008. The latter discusses also the relation
to the similar approximation scheme of Resumming the large-N approximation for
time evolving quantum systems, B. Mihaila, F. Cooper and J. F. Dawson, Phys. Rev.
D 63 (2001) 096003
•
 Far-from-equilibrium quantum fields and thermalisation are discussed in: Ther-
malization of Quantum Fields from Time-Reversal Invariant Evolution Equations,
J. Berges and J. Cox, Phys. Lett. B517 (2001) 369-374. Nonequilibrium time evolu-
tion of the spectral function in quantum field theory, G. Aarts and J. Berges, Phys.
Rev. D 64 (2001) 105010. Controlled nonperturbative dynamics of quantum fields
out of equilibrium, J. Berges, Nucl. Phys. A699 (2002) 847. Quantum dynamics of
phase transitions in broken symmetry φ 4 field theory, F. Cooper, J. F. Dawson and
B. Mihaila, Phys. Rev. D67 (2003) 056003. Thermalization of fermionic quantum
fields, J. Berges, Sz. Borsányi and J. Serreau, Nucl. Phys. B660 (2003) 52. Bose-
Einstein condensation without chemical potential, D. J. Bedingham, Phys. Rev. D
68 (2003) 105007. Quantum dynamics and thermalization for out-of-equilibrium
phi**4-theory, S. Juchem, W. Cassing and C. Greiner, Phys. Rev. D 69 (2004)
025006.
•
 The phenomenon of prethermalization is discussed in: Prethermalization,
J. Berges, Sz. Borsányi and C. Wetterich, Phys. Rev. Lett. in print, http://arXiv:hep-
ph/0403234
•
 Far-from-equilibrium dynamics of macroscopic fields with large fluctuations using
the 2PI 1/N expansion are discussed in: Parametric resonance in quantum field the-
ory, J. Berges and J. Serreau, Phys. Rev. Lett. 91 (2003) 111601. The leading-order
large-N description has been given in: Analytic and numerical study of preheating
Introduction to Nonequilibrium Quantum Field Theory
 129
dynamics, D. Boyanovsky, H. J. de Vega, R. Holman and J. F. Salgado, Phys. Rev.
D 54 (1996) 7570.
•
 A detailed discussion about Gaussian initial density matrices and dynamics can
be found in: Nonequilibrium dynamics of symmetry breaking in lambda Phi**4
field, F. Cooper, S. Habib, Y. Kluger and E. Mottola, Phys. Rev. D 55 (1997) 6471.
Gaussian dynamics for inhomogeneous fields are discussed in: Particle produc-
tion and effective thermalization in inhomogeneous mean field theory, G. Aarts
and J. Smit, Phys. Rev. D 61 (2000) 025002. Staying thermal with Hartree ensem-
ble approximations, M. Salle, J. Smit and J. C. Vink, Nucl. Phys. B 625 (2002)
495. Dynamical behavior of spatially inhomogeneous relativistic lambda phi**4
quantum field theory in the Hartree approximation, L. M. Bettencourt, K. Pao and
J. G. Sanderson, Phys. Rev. D 65 (2002) 025015. For a leading-order study includ-
ing fermions see: Nonequilibrium dynamics of fermions in a spatially homogeneous
scalar background field, J. Baacke, K. Heitmann and C. Pätzold, Phys. Rev. D 58
(1998) 125013.
•
 The renormalization of 2PI approximation schemes is discussed in: Renormal-
ization in self-consistent approximations schemes at finite temperature. I: The-
ory, H. van Hees and J. Knoll, Phys. Rev. D 65 (2002) 025010. Renormalization
of self-consistent approximation schemes II: Applications to the sunset diagram,
H. Van Hees and J. Knoll, Phys. Rev. D 65 (2002) 105005. Renormalization in
self-consistent approximation schemes at finite temperature III: Global symmetries,
H. van Hees and J. Knoll, Phys. Rev. D 66 (2002) 025028. Renormalizability of
Phi-derivable approximations in scalar phi**4 theory, J. P. Blaizot, E. Iancu and
U. Reinosa, Phys. Lett. B 568 (2003) 160. Renormalization of Φ-derivable approx-
imation schemes in scalar field theories, J.-P. Blaizot, E. Iancu and U. Reinosa,
Nucl. Phys. A 736 (2004) 149. Renormalizing the Schwinger-Dyson equations in
the auxiliary field formulation of lambda phi**4 field theory, F. Cooper, B. Mi-
haila and J. F. Dawson, http://arXiv:hep-ph/0407119. Renormalized thermodynam-
ics from the 2PI effective action, J. Berges, Sz. Borsanyi, U. Reinosa and J. Serreau,
http://arXiv:hep-ph/0409123.
•
 Classical aspects of nonequilibrium quantum fields and precision tests of the 2PI
1/N expansion are examined in: Classical aspects of quantum fields far from equi-
librium, G. Aarts and J. Berges, Phys. Rev. Lett. 88 (2002) 0416039. Nonequi-
librium quantum fields and the classical field theory limit, J. Berges, Nucl. Phys.
A 702 (2002) 351. Tachyonic preheating using 2PI - 1/N dynamics and the clas-
sical approximation, A. Arrizabalaga, J. Smit and A. Tranberg, http://arXiv:hep-
ph/0409177.
•
 Classical statistical field theory studies related to approximation schemes in QFT
can be found in: Exact and truncated dynamics in nonequilibrium field theory,
G. Aarts, G. F. Bonini and C. Wetterich, Phys. Rev. D 63 (2001) 025012. On
thermalization in classical scalar field theory, G. Aarts, G. F. Bonini, C. Wet-
terich, Nucl. Phys. B587 (2000) 403. Time evolution of correlation functions and
thermalization, G. F. Bonini and C. Wetterich, Phys. Rev. D 60 (1999) 105026.
Classical limit of time-dependent quantum field theory: A Schwinger-Dyson ap-
proach, F. Cooper, A. Khare and H. Rose, Phys. Lett. B 515 (2001) 463. K. Bla-
goev, F. Cooper, J. Dawson and B. Mihaila, Phys. Rev. D 64 (2001) 125003.
Introduction to Nonequilibrium Quantum Field Theory
 130
For diagrammatics in classical field theory see: Finiteness of hot classical scalar
field theory and the plasmon damping rate, G. Aarts and J. Smit, Phys. Lett. B
393 (1997) 395. Classical statistical mechanics and Landau damping, W. Buch-
müller and A. Jakovác, Phys. Lett. B 407 (1997) 39. Classical approximation for
time-dependent quantum field theory: Diagrammatic analysis for hot scalar fields,
G. Aarts and J. Smit, Nucl. Phys. B511 (1998) 451.
• The relation to kinetic equations is discussed in: Nonequilibrium Quantum Fields:
Closed Time Path Effective Action, Wigner Function And Boltzmann Equation,
E. Calzetta and B. L. Hu, Phys. Rev. D 37 (1988) 2878; Exact Conservation
Laws of the Gradient Expanded Kadanoff-Baym Equations, J. Knoll, Y. B. Ivanov
and D. N. Voskresensky, Annals Phys. 293 (2001) 126. Nonequilibrium quantum
fields with large fluctuations, J. Berges and M. M. Muller, in Progress in Nonequi-
librium Green’s Functions II, Eds. M. Bonitz and D. Semkat, World Scientific
(2003) [http://arXiv:hep-ph/0209026]. Cf. also Stochastic dynamics of correlations
in quantum field theory: From Schwinger-Dyson to Boltzmann-Langevin equation,
E. Calzetta and B. L. Hu, Phys. Rev. D 61 (2000) 025012.
• For a discussion of transport coefficients employing 2PI effective actions see: Hy-
drodynamic transport functions from quantum kinetic field theory, E. A. Calzetta,
B. L. Hu and S. A. Ramsey, Phys. Rev. D 61 (2000) 125013. Transport coefficients
from the 2PI effective action, G. Aarts and J. M. Martinez Resco, Phys. Rev. D 68
(2003) 085009. Shear viscosity in the O(N) model, G. Aarts and J. M. Martinez
Resco, JHEP 0402 (2004) 061. Transport coefficients at leading order: Kinetic
theory versus diagrams, G. D. Moore, in Strong and Electroweak Matter 2002,
ed. M.G. Schmidt (World Scientific, 2003), http://arXiv:hep-ph/0211281. Cf. also
Transport coefficients in high temperature gauge theories II: Beyond leading log,
P. Arnold, G. D. Moore and L. G. Yaffe, JHEP 0305 (2003) 051.
• For an application to photon production rates in a non-equilibrium medium see:
Out-of-equilibrium electromagnetic radiation, J. Serreau, JHEP 0405 (2004) 078.
Figures. Figs. 3–5: J. Berges, Nucl. Phys. A699 (2002) 847. Fig. 6 left: J. Berges
and J. Cox, Phys. Lett. B517 (2001) 369-374. Fig. 6 right: G. Aarts and J. Berges, Phys.
Rev. D 64 (2001) 105010. Figs. 7–10: J. Berges, Sz. Borsányi and C. Wetterich, Phys.
Rev. Lett. in print, http://arXiv:hep-ph/0403234. Figs. 11–14: J. Berges and J. Serreau,
Phys. Rev. Lett. 91 (2003) 111601. Figs. 16–18: G. Aarts and J. Berges, Phys. Rev. Lett.
88 (2002) 0416039.
Introduction to Nonequilibrium Quantum Field Theory
'''

nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

total_words = doc.split()
total_word_length = len(total_words)
print(total_word_length)

import string

table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in total_words]
# remove remaining tokens that are not alphabetic
total_words = [word for word in stripped if word.isalpha()]

total_word_length = len(total_words)
print(total_word_length)

total_words = [word for word in total_words if  len(word) > 4]

total_word_length = len(total_words)
print(total_word_length)


tf_score = {}
for each_word in total_words:
    each_word = each_word.replace('.', '')
    if each_word not in stop_words:
        if each_word in tf_score:
            tf_score[each_word] += 1
        else:
            tf_score[each_word] = 1

# Dividing by total_word_length for each dictionary element
tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())


def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))


total_sentences = tokenize.sent_tokenize(doc)
total_sent_len = len(total_sentences)

idf_score = {}
for each_word in total_words:
    each_word = each_word.replace('.', '')
    if each_word not in stop_words:
        if each_word in idf_score:
            idf_score[each_word] = check_sent(each_word, total_sentences)
        else:
            idf_score[each_word] = 1

# Performing a log and divide
idf_score.update((x, math.log(int(total_sent_len)/y))
                 for x, y in idf_score.items())

tf_idf_score = {key: tf_score[key] *
                idf_score.get(key, 0) for key in tf_score.keys()}


def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(),
                         key=itemgetter(1), reverse=True)[:n])
    return result


print(get_top_n(tf_idf_score, 40))

