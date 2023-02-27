from typing import Callable
import numpy as np

# Define global constants to speed up computation
pi = np.pi
pi2 = 2 * np.pi
pid2 = np.pi / 2
pi3d2 = 3 * np.pi / 2

def idx(i: int, j: int) -> int:
    '''Return the index of the i-th particle's j-th coordinate.'''
    return j + 2*i

def rotate_coord(x, y, cth, sth):
    '''Rotate point (x,y) by angle theta corresponding to cos(theta) = cth
    and sin(theta) = sth.'''
    return x*cth - y*sth, x*sth + y*cth

class Potential():
    '''Class for pairwise interaction potentials between particles.'''

    def __init__(self):
        pass

    def pair_energy(self, d: float, i: int, j: int):
        '''Return the interaction energy between particles i and j'''
        raise NotImplementedError("Pair energy only implemented for specific subclasses!")

    def grad_factor(self, d: float):
        '''Return the coefficient in the energy gradient of two particles
        separated by distance d.'''
        raise NotImplementedError("Grad factor only implemented for specific subclasses!")


class BF(Potential):
    '''
    Bladon-Frenkel potential from Bladon and Frenkel, PRL 1995.
    '''
    def __init__(self, epsilon: float, delsig: float, c0: float = 1.5):
        self.epsilon = epsilon
        self.sigma = 2 / 3
        self.delta = delsig * self.sigma
        self.coeff = 4 * epsilon / self.delta**2
        self.c0 = c0
        self.coeffcore = -12 * c0**12
        self.sig3d2 = self.sigma * 3 / 2
        self.deld2 = self.delta/2
        self.coeff2 = 2 * self.coeff
        self.bound1 = self.sig3d2-self.deld2
        self.bound2 = self.sig3d2+self.deld2

    def pair_energy(self, d: float, i: int, j: int):
        '''Return the interaction energy between particles i and j'''
        assert d > 0,"Invalid distance for particles %d & %d" % (i,j)
        if (d >= 3/2*self.sigma - self.deld2 and d <= self.sig3d2 + self.deld2):
            return -self.epsilon + self.coeff*(d-self.sig3d2)**2
        if d <= self.sig3d2 - self.deld2:
            return (self.c0/d)**12 - (self.c0/(self.sig3d2 - self.deld2))**12
        return 0.0
  
    def grad_factor(self, d):
        '''Return the coefficient in the energy gradient of two particles
        separated by distance d.'''
        if d < self.bound1:
            return self.coeffcore/d**14
        if (d >= self.bound1 and d <= self.bound2):
            return self.coeff2*(d - self.sig3d2)/d
        return 0.0


class LJ(Potential):
    '''
    Lennard-Jones potential
    V(r) = epsilon * ((a0/r)**12 - 2*(a0/r)**6)
         = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    where sigma = a0/s**(1/6).
    '''
    def __init__(self, epsilon: float, a0: float = 1.0, cutoff: float = 2.5):
        self.epsilon = epsilon
        self.coeff = 4 * epsilon
        self.cutoff = cutoff
        self.en_s6 = (a0**6 / 2) * self.coeff
        self.en_s12 = (a0**12 / 4) * self.coeff
        self.g_coeff = - 12 * self.epsilon

    def pair_energy(self, d: float, i: int, j: int):
        '''Return the interaction energy between particles i and j'''
        assert d > 0, "Invalid distance for particles %d & %d" % (i, j)
        if d <= self.cutoff:
            return self.en_s12/d**12 - self.en_s6/d**6
        return 0.0
 
    def grad_factor(self, d):
        '''Return the coefficient in the energy gradient of two particles
        separated by distance d.'''
        if d < self.cutoff:
            return self.g_coeff*(1/d**14 - 1/d**8)
        return 0.0
    

class Crystal():
    '''Class for a(n approximately triangular) lattice of particles on a conic surface.'''

    def __init__(self, n_init: int, n_rings: int, ctheta: float, fixring: bool = False, stagger: float = 0.5):
        self.n_init = n_init # Number of atoms in the middle initial ring 
        self.n_rings = n_rings
        self.r_init = n_init / ctheta
        self.ctheta = ctheta
        self.cthetad2 = ctheta / 2
        self.stagger = stagger
        self.fixring = fixring
        self.cth = np.cos(ctheta)
        self.sth = np.sin(ctheta)
        self.npr = [int(round(ctheta*(self.r_init + np.sqrt(3)/2*k)))
                    for k in range(-n_rings, n_rings+1)]
        self.initialize_ring()
        self.init_pos = self.pos.copy()

    def add_ring(self, k: int, theta_max: float):
        '''Add a ring of particles at distance k * sqrt(3)/2 from the center.'''
        n_particles = self.npr[self.n_rings + k]
        dtheta = self.ctheta / n_particles
        r = self.r_init + k * np.sqrt(3)/2
        for i in range(n_particles):
            self.polar_pos += [r, theta_max - (i + self.stagger*(k%2)) * dtheta]

    def initialize_ring(self):
        '''Initialize positions of (1 + 2 * n_rings) rings of particles.'''
        theta_max = np.pi / 2 + 0.5 * self.ctheta
        self.polar_pos = []
        for k in range(-self.n_rings, self.n_rings+1):
            if k == 0: self.init_index = len(self.polar_pos) // 2
            self.add_ring(k, theta_max)
        self.N = len(self.polar_pos) // 2
        self.pos = np.zeros(len(self.polar_pos))
        for i in range(self.N): # Convert to Cartesian
            self.pos[idx(i, 0)] = (self.polar_pos[idx(i, 0)] *
                                   np.cos(self.polar_pos[idx(i, 1)]))
            self.pos[idx(i, 1)] = (self.polar_pos[idx(i, 0)] *
                                   np.sin(self.polar_pos[idx(i, 1)]))

    def implement_pbc(self):
        '''Implement periodic boundary conditions on the crystal along the
        azimuthal direction of the cone.'''
        for i in range(self.N):
            x_init, y_init, ang1 = self.ind_to_coord(i)
            # if out of bounds on the right side
            if ang1 < ((pid2 - self.cthetad2) % pi2):
                if self.ctheta < pi or (self.ctheta > pi and ang1 > pi3d2):
                    # rotate by an angle ctheta to bring it back within bounds
                    self.pos[idx(i, 0)], self.pos[idx(i, 1)] = rotate_coord(
                        x_init, y_init, self.cth, self.sth)
            # if out of bounds on the left side
            if pi3d2 > ang1 >= pid2 + self.cthetad2:
                # rotate by an angle -ctheta to bring it back within bounds
                self.pos[idx(i, 0)], self.pos[idx(i, 1)] = rotate_coord(
                    x_init, y_init, self.cth, -self.sth)

    def update_energy(self, potential: BF, all_geodesics: bool = False):
        '''Calculate the energy of the crystal given its current positions.'''
        self.implement_pbc()
        self.energy = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.add_pair_energy(i, j, potential.pair_energy)
                if all_geodesics:
                    self.add_all_geodesics(i, j, potential, calc_energy=True)

    def add_pair_energy(self, i: int, j: int, pair_energy: Callable):
        '''Add the interaction energy of particles i, j to the total energy.'''
        x1, y1, ang1 = self.ind_to_coord(i)
        x2, y2, ang2 = self.ind_to_coord(j)
        if np.abs(ang2 - ang1) <= pi:
            d = np.sqrt((x1-x2)**2 +(y1-y2)**2)
            self.energy += pair_energy(d, i, j)
        if ang2 - ang1 > self.ctheta - pi:
            xr, yr = x2*self.cth + y2*self.sth, - x2*self.sth + y2*self.cth
            d = np.sqrt((x1-xr)**2 + (y1-yr)**2)
            self.energy += pair_energy(d, i, j)
        if ang2 - ang1 < -(self.ctheta - pi):
            xl, yl = x2*self.cth - y2*self.sth, x2*self.sth + y2*self.cth
            d = np.sqrt((x1-xl)**2 + (y1-yl)**2)
            self.energy += pair_energy(d, i, j)

    def update_grad(self, potential: Potential, all_geodesics: bool = False):
        '''Calculate the gradient of the particles given current positions.'''
        self.grad = np.zeros(2 * self.N)
        for i in range(self.N):
            for j in range(self.N):
                if j != i:
                    self.gradfxn(i, j, potential)
                    if all_geodesics:
                        self.add_all_geodesics(i, j, potential)
        if not self.fixring: # Pin a point so that crystal cannot translate
            i = 0 # self.init_index + self.n_init // 2
            self.grad[idx(i, 0)], self.grad[idx(i, 1)] = 0, 0
        else: # Pin the entire initial ring
            for i in range(self.init_index, self.init_index + self.n_init):
                self.grad[idx(i, 0)], self.grad[idx(i, 1)] = 0, 0

    def update_pair_grad(self, factor, i, j, x, y, cth, sth):
        '''Add gradient contributions from particles i and, j or j's image.'''
        self.grad[idx(i, 0)] += factor * x
        self.grad[idx(i, 1)] += factor * y
        self.grad[idx(j, 0)] -= factor * (x*cth - y*sth)
        self.grad[idx(j, 1)] -= factor * (x*sth + y*cth)

    def gradfxn(self, i, j, potential: BF):
        '''
        Calculates gradients for particles i, j along their connecting
        geodesics. On the isometric mapping of an conic surface, geodesics
        between two particles consist of any straight line connecting
        a point to another point or its image points in the polar plane. 
        '''
        x1, y1, ang1 = self.ind_to_coord(i)
        x2, y2, ang2 = self.ind_to_coord(j)
        # Include geodesic between the two original pts
        if np.abs(ang2 - ang1) <= pi:
            d = np.sqrt((x1-x2)**2+(y1-y2)**2)
            factor = potential.grad_factor(d)
            self.update_pair_grad(factor, i, j, x1-x2, y1-y2, 1., 0.)
        # Check for geodesic between an original pt and a CCW rotated image pt
        if ang2 - ang1 > self.ctheta - pi: 
            xr, yr = rotate_coord(x2, y2, self.cth, -self.sth) 
            d = np.sqrt((x1-xr)**2 + (y1-yr)**2)
            factor = potential.grad_factor(d)
            self.update_pair_grad(factor, i, j, x1-xr, y1-yr, self.cth,
                                  self.sth)
        # Check for geodesic between an original pt and a CW rotated image pt
        if ang2 - ang1 < -(self.ctheta - pi):
            xl, yl = rotate_coord(x2, y2, self.cth, self.sth)
            d = np.sqrt((x1-xl)**2 +(y1-yl)**2)
            factor = potential.grad_factor(d)
            self.update_pair_grad(factor, i, j, x1-xl, y1-yl, self.cth,
                                  -self.sth)

    def ind_to_coord(self, i):
        '''Convert particle index to its position and angle.'''
        x1, y1 = self.pos[idx(i, 0)], self.pos[idx(i, 1)]
        ang1 = np.angle(x1+1j*y1) % pi2
        return x1, y1, ang1

    def add_all_geodesics(self, i, j, potential: BF, calc_energy: bool = False):
        '''Add all higher order geodesics from remaining image points, which
        are non-negligible for particles near the cone tip.'''
        x1, y1, ang1 = self.ind_to_coord(i)
        x2, y2, ang2 = self.ind_to_coord(j)
        nG = np.floor((ang2 - ang1 + pi)/self.ctheta)
        for n in range(2, nG):
            cnth, snth = np.cos(n*self.ctheta), np.sin(n*self.ctheta)
            xr, yr = rotate_coord(x2, y2, cnth, -snth)
            d = np.sqrt((x1-xr)**2 +(y1-yr)**2)
            if calc_energy:
                self.energy += potential.pair_energy(d, i, j)
                continue
            factor = potential.grad_factor(d)
            self.update_pair_grad(factor, i, j, x1-xr, y1-yr, cnth, snth)
        nG = np.floor((-ang2 + ang1 + pi)/self.ctheta)
        for n in range(2, nG):
            cnth, snth = np.cos(n*self.ctheta), np.sin(n*self.ctheta)
            xl, yl = rotate_coord(x2, y2, cnth, snth)
            d = np.sqrt((x1-xl)**2 +(y1-yl)**2)
            if calc_energy:
                self.energy += potential.pair_energy(d, i, j)
                continue
            factor = potential.grad_factor(d)
            self.update_pair_grad(factor, i, j, x1-xl, y1-yl, cnth, -snth)


class FIRE():
    '''Class for fast inertial relaxation engine (FIRE) for structure optimization.'''

    def __init__(self, alpha_s: float, falph: float, dtmax: float, finc: float,
                 fdec: float, dumpstep: int, crit: float, nmin: int = 5):
        self.alpha_s = alpha_s
        self.falph = falph
        self.dtmax = dtmax
        self.finc = finc
        self.fdec = fdec
        self.dumpstep = dumpstep
        self.crit = crit
        self.nmin = nmin
        self.alpha = alpha_s
        self.dt = dtmax * 0.1
        self.npos = 0

    def step(self, latt: Crystal, potential: Potential, v, timestep):
        '''
        Perform one step of FIRE optimization:
        Update positions with velocity verlet, then forces and velocities
        Can try velocity mixing in the middle of calculating velocities
        (as in Guenole (2020)) if this becomes unstable.
        '''
        v, latt = velocity_verlet(latt, potential, v, self.dt)
        p, normV, normF = 0, 0, 0
        for i in range(latt.N): # calculate power
            for k in range(2):
                p -= v[idx(i, k)] * latt.grad[idx(i, k)] # power = v dot F
                normV += v[idx(i, k)] * v[idx(i, k)]
                normF += latt.grad[idx(i, k)] * latt.grad[idx(i, k)]
        normV, normF = np.sqrt(normV), np.sqrt(normF)
        # check power and change parameters accordingly
        if p > 0:
            for i in range(latt.N):
                for k in range(2):
                    v[idx(i, k)] -= (self.alpha * v[idx(i, k)] +
                                     self.alpha * latt.grad[idx(i, k)] * 
                                     normV/normF)
                if (self.npos > self.nmin and timestep > 10):
                    # increase timestep towards max allowed
                    self.dt = min(self.dt*self.finc, self.dtmax) 
                    self.alpha *= self.falph # decrease 'steering' (damping)
                self.npos += 1
        if p <= 0: # correct uphill motion, suggested by LAMMPS (Guenole 2020). 
            for i in range(latt.N):
                for k in range(2):
                    latt.pos[idx(i, k)] -= .5 * self.dt * v[idx(i, k)]
            v = 0.0 * v # reset velocity 
            if timestep > 10: # For stability, fix parameters for first 10 steps
                self.dt *= self.fdec # decrease timestep
                self.alpha = self.alpha_s # reset mixing to have more steering
            self.npos = 0 # reset steps passed since power was positive
        return v, latt

    def run(self, latt: Crystal, potential: Potential):
        '''Run FIRE optimization until convergence.'''
        v = np.zeros(latt.N * 2)
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

def velocity_verlet(latt: Crystal, potential: BF, v, dt):
    '''Run one step of velocity verlet: updates positions then velocities.'''
    for i in range(latt.N):
        for k in range(2):
            # could insert 1/m in front of F
            latt.pos[idx(i, k)] += (v[idx(i, k)]*dt
                                    - 0.5*latt.grad[idx(i, k)]*dt**2)
            latt.implement_pbc()
            v[idx(i, k)] -= 0.5*latt.grad[idx(i, k)]*dt
    latt.update_grad(potential)
    for i in range(latt.N):
        for k in range(2):
            v[idx(i, k)] -= 0.5*latt.grad[idx(i, k)]*dt
    return v, latt
