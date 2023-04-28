"""
Program for structural relaxation of particles on conic surfaces.

Classes:

    Potential -- Abstract class for pairwise interaction potentials between
    particles.
    Crystal -- Class for an (approximately six-fold coordinated) lattice of
    particles on a conic surface.
    FIRE -- Class for the Fast Inertial Relaxation Engine algorithm.

Helper functions:

    rotate_coord -- Rotate point (x,y) about the origin by a given angle.
    velocity_verlet -- Perform a single step of velocity Verlet integration.
"""

from typing import Callable, List, Tuple
from dataclasses import dataclass, field
import numpy as np

# Define global constants to speed up computation
pi = np.pi
pi2 = 2 * np.pi
pid2 = np.pi / 2
pi3d2 = 3 * np.pi / 2


class Potential():
    '''Class for pairwise interaction potentials between particles. Subclass
    this to implement specific potentials.'''

    def __init__(self):
        pass

    def pair_energy(self, d: float, i: int, j: int):
        '''Return the interaction energy between particles i and j'''
        raise NotImplementedError("Pair energy only implemented for concrete subclasses!")

    def grad_factor(self, d: float):
        '''Return the coefficient in the energy gradient of two particles
        separated by distance d.'''
        raise NotImplementedError("Grad factor only implemented for concrete subclasses!")


class BF(Potential):
    '''
    Potential inspired by Bladon and Frenkel, PRL 1995.
    V(r < a0 - delta/2)
        = c0*(1/r^12 - 1/(a0 - delta/2)^12)
    V(a0 - delta/2 <= r <= a0 + delta/2)
        = -epsilon + 4*epsilon*(r - a0)^2/delta^2
    V(r > a0 + delta/2) = 0
    where a0 = 3*sigma/2 where sigma is the original parameter in literature.
    '''
    def __init__(self, epsilon: float, delsig: float, c0: float = 1.5) -> None:
        self.epsilon = epsilon # Depth of the attractive potential well
        self.c0 = c0 # Repulsive core strength
        self.sigma = 2 / 3 # Convention; sigma = 2/3 * distance at well minimum
        self.delta = delsig * self.sigma # Width of the attractive well
        self.coeff = 4 * epsilon / self.delta**2
        self.coeffcore = -12 * c0**12
        self.sig3d2 = self.sigma * 3 / 2
        self.deld2 = self.delta/2
        self.coeff2 = 2 * self.coeff
        self.bound1 = self.sig3d2-self.deld2
        self.bound2 = self.sig3d2+self.deld2

    def pair_energy(self, d: float, i: int, j: int) -> float:
        '''Return the interaction energy between particles i and j'''
        assert d > 0, "Invalid distance for particles %d & %d" % (i, j)
        if (self.sig3d2 - self.deld2 <= d <= self.sig3d2 + self.deld2):
            return -self.epsilon + self.coeff*(d-self.sig3d2)**2
        if d <= self.sig3d2 - self.deld2:
            return (self.c0/d)**12 - (self.c0/(self.sig3d2 - self.deld2))**12
        return 0.0

    def grad_factor(self, d: float) -> float:
        '''Return the coefficient in the energy gradient of two particles
        separated by distance d.'''
        if d < self.bound1:
            return self.coeffcore/d**14
        if (self.bound1 <= d <= self.bound2):
            return self.coeff2*(d - self.sig3d2)/d
        return 0.0


class LJ(Potential):
    '''
    Lennard-Jones potential
    V(r) = epsilon * ((a0/r)**12 - 2*(a0/r)**6)
         = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    where sigma = a0/2**(1/6).
    '''
    def __init__(self, epsilon: float, a0: float = 1.0,
                 cutoff: float = 2.5) -> None:
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.coeff = 4 * epsilon
        self.en_s6 = (a0**6 / 2) * self.coeff
        self.en_s12 = (a0**12 / 4) * self.coeff
        self.g_coeff = - 12 * self.epsilon

    def pair_energy(self, d: float, i: int, j: int) -> float:
        '''Return the interaction energy between particles i and j'''
        assert d > 0, "Invalid distance for particles %d & %d" % (i, j)
        if d <= self.cutoff:
            return self.en_s12/d**12 - self.en_s6/d**6
        return 0.0

    def grad_factor(self, d) -> float:
        '''Return the coefficient in the energy gradient of two particles
        separated by distance d.'''
        if d < self.cutoff:
            return self.g_coeff*(1/d**14 - 1/d**8)
        return 0.0


@dataclass
class Crystal:
    '''Class for an (approximately six-fold coordinated) lattice of particles on a conic surface.'''
    n_init: int # Number of atoms in the commensurate seeding ring
    n_rings: int # Number of pairs of rings above and below the seeding ring
    ctheta: float # Sector angle of the rolled-out conic surface
    fix_ring: bool = False # Fix the initial ring in place
    stagger: float = 0.5 # Stagger neighboring rings in initial configuration
    # by this fraction of the lattice constant
    init_pos: np.ndarray = field(init=False) # Initial particle positions
    r_init: float = field(init=False) # Distance of seeding ring from the apex
    cth: float = field(init=False) # Cosine of the sector angle
    sth: float = field(init=False) # Sine of the sector angle
    cthetad2: float = field(init=False) # Half the sector angle
    npr: List[int] = field(init=False) # Number of particles in each ring
    N: int = field(init=False) # Total number of particles
    energy: float = field(init=False) # Total energy of the system
    grad: np.ndarray = field(init=False) # Gradient of the energy

    def __post_init__(self) -> None:
        self.r_init = self.n_init / self.ctheta
        self.cthetad2 = self.ctheta / 2
        self.cth = np.cos(self.ctheta)
        self.sth = np.sin(self.ctheta)
        self.npr = [int(round(self.ctheta*(self.r_init + np.sqrt(3)/2*k)))
                    for k in range(-self.n_rings, self.n_rings+1)]
        self.N = sum(self.npr)
        self.initialize_ring()
        self.init_pos = self.pos.copy()

    def add_ring(self, k: int, theta_max: float) -> None:
        '''Add a ring of particles at distance k * sqrt(3)/2 from the center.
        Side effects: adds k-th pair of rings to self.polar_pos.'''
        n_particles = self.npr[self.n_rings + k]
        dtheta = self.ctheta / n_particles
        r = self.r_init + k * np.sqrt(3)/2
        for i in range(n_particles):
            self.polar_pos += [r, theta_max - (i + self.stagger*(k%2)) * dtheta]

    def initialize_ring(self) -> None:
        '''Initialize positions of (1 + 2 * n_rings) rings of particles.
        Side effects: sets self.polar_pos and self.pos.'''
        theta_max = pid2 + self.cthetad2
        self.polar_pos = []
        for k in range(-self.n_rings, self.n_rings+1):
            if k == 0:
                self.init_index = len(self.polar_pos) // 2
            self.add_ring(k, theta_max)
        self.polar_pos = np.array(self.polar_pos).reshape((self.N, 2))
        self.pos = np.zeros((self.N, 2))
        for i in range(self.N): # Convert to Cartesian
            self.pos[i, 0] = (self.polar_pos[i, 0] *
                              np.cos(self.polar_pos[i, 1]))
            self.pos[i, 1] = (self.polar_pos[i, 0] *
                              np.sin(self.polar_pos[i, 1]))

    def implement_pbc(self) -> None:
        '''Implement periodic boundary conditions on the crystal along the
        azimuthal direction of the cone.
        Side effects: modifies self.pos to ensure all particles are within the
        polar sector from 0 to ctheta.
        '''
        for i in range(self.N):
            x_init, y_init, ang1 = self.ind_to_coord(i)
            # if out of bounds on the right side
            if ang1 < ((pid2 - self.cthetad2) % pi2):
                if self.ctheta < pi or (self.ctheta > pi and ang1 > pi3d2):
                    # rotate by an angle ctheta to bring it back within bounds
                    self.pos[i, 0], self.pos[i, 1] = rotate_coord(
                        x_init, y_init, self.cth, self.sth)
            # if out of bounds on the left side
            if pi3d2 > ang1 >= pid2 + self.cthetad2:
                # rotate by an angle -ctheta to bring it back within bounds
                self.pos[i, 0], self.pos[i, 1] = rotate_coord(
                    x_init, y_init, self.cth, -self.sth)

    def update_energy(self, pot: Potential,
                      all_geodesics: bool = False) -> None:
        '''Calculate the energy of the crystal given its current positions.
        Side effects: sets self.energy to the total energy of the particles.'''
        self.implement_pbc()
        self.energy = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.add_pair_energy(i, j, pot.pair_energy)
                if all_geodesics:
                    self.add_all_geodesics(i, j, pot, calc_energy=True)

    def add_pair_energy(self, i: int, j: int, pair_energy: Callable) -> None:
        '''Add the interaction energy of particles i, j to the total energy.
        Side effects: modifies self.energy.'''
        x1, y1, ang1 = self.ind_to_coord(i)
        x2, y2, ang2 = self.ind_to_coord(j)
        ang_diff = ang2 - ang1
        if np.abs(ang_diff) <= pi:
            self.energy += pair_energy(dist(x1-x2, y1-y2), i, j)
        if ang_diff > self.ctheta - pi:
            xr, yr = rotate_coord(x2, y2, self.cth, -self.sth)
            self.energy += pair_energy(dist(x1-xr, y1-yr), i, j)
        if ang_diff < -(self.ctheta - pi):
            xl, yl = rotate_coord(x2, y2, self.cth, self.sth)
            self.energy += pair_energy(dist(x1-xl, y1-yl), i, j)

    def update_grad(self, pot: Potential, all_geodesics: bool = False) -> None:
        '''Calculate the gradient of the particles given current positions.
        Side effects: sets self.grad to the gradient of the particles.'''
        self.grad = np.zeros((self.N, 2))
        for i in range(self.N):
            for j in range(self.N):
                if j != i:
                    self.gradfxn(i, j, pot)
                    if all_geodesics:
                        self.add_all_geodesics(i, j, pot)
        if not self.fix_ring: # Pin a point so that crystal cannot translate
            i = 0 # self.init_index + self.n_init // 2
            self.grad[i] = 0, 0
        else: # Pin the entire initial ring
            for i in range(self.init_index, self.init_index + self.n_init):
                self.grad[i] = 0, 0

    def update_pair_grad(self, i, j, x, y, cth, sth, pot: Potential) -> None:
        '''Add gradient contributions from particles i and, j or j's image.
        Side effects: modifies self.grad.'''
        d = dist(x, y)
        factor = pot.grad_factor(d)
        self.grad[i] += factor * x, factor * y
        # rotate opposing gradient from image frame to the original frame
        x, y = rotate_coord(x, y, cth, sth)
        self.grad[j] -= factor * x, factor * y

    def gradfxn(self, i, j, pot: Potential) -> None:
        '''
        Calculates gradients for particles i, j along their connecting
        geodesics. In the isometric mapping of an conic surface, geodesics
        between two particles consist of any straight line connecting
        a point to another point or its image points in the polar plane.
        Side effects: modifies self.grad.
        '''
        x1, y1, ang1 = self.ind_to_coord(i)
        x2, y2, ang2 = self.ind_to_coord(j)
        ang_diff = ang2 - ang1
        # Include geodesic between the two original pts
        if np.abs(ang_diff) <= pi:
            self.update_pair_grad(i, j, x1-x2, y1-y2, 1., 0., pot)
        # Check for geodesic between an original pt and a CW rotated image pt
        if ang_diff > self.ctheta - pi:
            xr, yr = rotate_coord(x2, y2, self.cth, -self.sth)
            self.update_pair_grad(i, j, x1-xr, y1-yr, self.cth, self.sth, pot)
        # Check for geodesic between an original pt and a CCW rotated image pt
        if ang_diff < -(self.ctheta - pi):
            xl, yl = rotate_coord(x2, y2, self.cth, self.sth)
            self.update_pair_grad(i, j, x1-xl, y1-yl, self.cth, -self.sth, pot)

    def ind_to_coord(self, i) -> Tuple[float, float, float]:
        '''Convert particle index to its position and angle.'''
        x1, y1 = self.pos[i]
        ang1 = np.angle(x1 + 1j*y1) % pi2
        return x1, y1, ang1

    def add_all_geodesics(self, i, j, pot: Potential,
                          calc_energy: bool = False) -> None:
        '''Add all higher order geodesics from remaining image points, which
        are non-negligible for particles near the cone tip.
        Side effects: modifies self.energy or self.grad.'''
        x1, y1, ang1 = self.ind_to_coord(i)
        x2, y2, ang2 = self.ind_to_coord(j)
        nG = np.floor((ang2 - ang1 + pi)/self.ctheta)
        for n in range(2, nG):
            cnth, snth = np.cos(n*self.ctheta), np.sin(n*self.ctheta)
            xr, yr = rotate_coord(x2, y2, cnth, -snth)
            if calc_energy:
                d = dist(x1-xr, y1-yr)
                self.energy += pot.pair_energy(d, i, j)
                continue
            self.update_pair_grad(i, j, x1-xr, y1-yr, cnth, snth, pot)
        nG = np.floor((-ang2 + ang1 + pi)/self.ctheta)
        for n in range(2, nG):
            cnth, snth = np.cos(n*self.ctheta), np.sin(n*self.ctheta)
            xl, yl = rotate_coord(x2, y2, cnth, snth)
            if calc_energy:
                d = dist(x1-xl, y1-yl)
                self.energy += pot.pair_energy(d, i, j)
                continue
            self.update_pair_grad(i, j, x1-xl, y1-yl, cnth, -snth, pot)


@dataclass
class FIRE:
    '''Class for fast inertial relaxation engine (FIRE) for structure optimization.'''
    alpha_s: float # Initial step size
    falph: float # Step size decrease factor
    dtmax: float # Maximum time step
    finc: float # Time step increase factor
    fdec: float # Time step decrease factor
    dumpstep: int # Number of steps between dumping data
    crit: float # Convergence criterion
    nmin: int = 5 # Minimum number of steps before increasing time step
    npos: int = 0 # Number of steps since last time step increase
    alpha: float = field(init=False) # Current step size
    dt: float = field(init=False) # Current time step

    def __post_init__(self) -> None:
        '''Set initial step size and time step.'''
        self.alpha = self.alpha_s
        self.dt = self.dtmax * 0.1

    def step(self, latt: Crystal, potential: Potential, v: np.ndarray,
             timestep: int) -> Tuple[np.ndarray, Crystal]:
        '''
        Perform one step of FIRE optimization:
        Update positions with velocity verlet, then forces and velocities
        Can try velocity mixing in the middle of calculating velocities
        (as in Guenole (2020)) if this becomes unstable.
        Side effects: modifies latt.pos, latt.grad, v, self.alpha, self.dt.
        '''
        v, latt = velocity_verlet(latt, potential, v, self.dt)
        p = -np.sum(v * latt.grad) # power = v dot F
        normV = np.sum(v**2)
        normF = np.sum(latt.grad**2)
        normV, normF = np.sqrt(normV), np.sqrt(normF)
        # check power and change parameters accordingly
        if p > 0:
            v -= (self.alpha*v + self.alpha*latt.grad*normV/normF)
            if (self.npos > self.nmin and timestep > 10):
                # increase timestep towards max allowed
                self.dt = min(self.dt*self.finc, self.dtmax)
                self.alpha *= self.falph # decrease 'steering' (damping)
            self.npos += 1
        if p <= 0: # correct uphill motion, suggested by LAMMPS (Guenole 2020)
            latt.pos -= .5 * self.dt * v
            v = 0.0 * v # reset velocity
            if timestep > 10: # For stability, fix parameters for first 10 steps
                self.dt *= self.fdec # decrease timestep
                self.alpha = self.alpha_s # reset mixing to have more steering
            self.npos = 0 # reset steps passed since power was positive
        return v, latt

    def run(self, latt: Crystal, potential: Potential) -> Tuple:
        '''Run FIRE optimization until convergence.
        Side effects: modifies latt.pos, latt.grad, latt.energy.'''
        v = np.zeros((latt.N, 2))
        latt.update_grad(potential)
        timestep, energies, positions = 0, [], []
        while np.max(np.abs(latt.grad)) > self.crit: # lenient criteria
            v, latt = self.step(latt, potential, v, timestep)
            if timestep % self.dumpstep == 0:
                latt.update_energy(potential)
                print('Energy at step ', timestep, ': ', latt.energy)
                energies.append([timestep, latt.energy])
                positions.append(latt.pos.copy())
            timestep += 1
        return energies, positions

def rotate_coord(x: float, y: float, cth: float, sth: float
                 ) -> Tuple[float, float]:
    '''Rotate point (x,y) by angle theta corresponding to cos(theta) = cth
    and sin(theta) = sth.'''
    return (x*cth - y*sth, x*sth + y*cth)

def dist(dx: float, dy: float) -> float:
    '''Return the Euclidean length of separation vector (dx, dy).'''
    return np.sqrt(dx**2 + dy**2)

def velocity_verlet(latt: Crystal, potential: BF, v, dt) -> Tuple[np.ndarray, Crystal]:
    '''Run one step of velocity verlet: updates positions then velocities.'''
    latt.pos += (v*dt - 0.5*latt.grad*dt**2)
    latt.implement_pbc()
    v -= 0.5*latt.grad*dt
    latt.update_grad(potential)
    v -= 0.5*latt.grad*dt
    return v, latt
