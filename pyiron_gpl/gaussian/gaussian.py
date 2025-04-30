# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import subprocess
import re
import pandas
import numpy as np
import matplotlib.pyplot as pt

from pyiron_base import GenericParameters
from pyiron_atomistics.dft.job.generic import GenericDFTJob
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron.atomistics.job.atomistic import Trajectory
from pyiron_snippets.import_alarm import ImportAlarm

from ase.units import Bohr, Ha, _me, _amu

try:
    from iodata import load_one, load_many
    import_alarm = ImportAlarm()
except ImportError:
    import_alarm = ImportAlarm(
        "Gaussian relies on the IOData package, but this is unavailable. Please ensure your python environment contains it."
    )


__author__ = "Jan Janssen, Sander Borgmans"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH - " \
                "- Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = ""
__email__ = ""
__status__ = "trial"
__date__ = "Aug 27, 2019"


class Gaussian(GenericDFTJob):
    @import_alarm
    def __init__(self, project, job_name):
        super(Gaussian, self).__init__(project, job_name)
        self.__name__ = "Gaussian"
        self._executable_activate(enforce=True)
        self.input = GaussianInput()


    def write_input(self):
        input_dict = {'mem': self.server.memory_limit, # per core memory
                      'cores': self.server.cores,
                      'verbosity': self.input['verbosity'],
                      'lot': self.input['lot'],
                      'basis_set': self.input['basis_set'],
                      'jobtype' : self.input['jobtype'],
                      'settings' : self.input['settings'],
                      'title' : self.input['title'],
                      'spin_mult': self.input['spin_mult'],
                      'charge': self.input['charge'],
                      'suffix': self.input['suffix'],
                      'bsse_idx': self.input['bsse_idx'],
                      'spin_orbit_states': self.input['spin_orbit_states'],
                      'symbols': self.structure.get_chemical_symbols().tolist(),
                      'pos': self.structure.positions
                      }
        write_input(input_dict=input_dict, working_directory=self.working_directory)


    def collect_output(self):
        output_dict = _collect_output(output_file=os.path.join(self.working_directory, 'input.fchk'))
        with self.project_hdf5.open("output") as hdf5_output:
            for k, v in output_dict.items():
                hdf5_output[k] = v


    def to_hdf(self, hdf=None, group_name=None):
        super(Gaussian, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)
            self.input.to_hdf(hdf5_input)


    def from_hdf(self, hdf=None, group_name=None):
        super(Gaussian, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.structure = Atoms().from_hdf(hdf5_input)


    def log(self):
        with open(os.path.join(self.working_directory, 'input.log')) as f:
            print(f.read())


    def calc_minimize(self, electronic_steps=None, ionic_steps=None, algorithm=None, ionic_forces=None):
        """
            Function to setup the hamiltonian to perform ionic relaxations using DFT. The convergence goal can be set using
            either the iconic_energy as an limit for fluctuations in energy or the iconic_forces.

            **Arguments**

                algorithm: SCF algorithm
                electronic_steps (int): maximum number of electronic steps per electronic convergence
                ionic_steps (int): maximum number of ionic steps
                ionic_forces ('tight' or 'verytight'): convergence criterium for Berny opt (optional)
        """
        settings = {}
        opt_settings = []

        if electronic_steps is not None:
            if not 'SCF' in settings:
                settings['SCF'] = []
            settings['SCF'].append("MaxCycle={}".format(electronic_steps))

        if ionic_steps is not None:
            opt_settings.append("MaxCycles={}".format(ionic_steps))

        if algorithm is not None:
            if not 'SCF' in settings:
                settings['SCF'] = []
            settings['SCF'].append(algorithm)

        if ionic_forces is not None:
            assert isinstance(ionic_forces,str)
            opt_settings.append(ionic_forces)

        self.input['jobtype'] = 'opt' + '({})'.format(",".join(opt_settings))*(len(opt_settings)>0)
        if not isinstance(self.input['settings'],dict):
            self.input['settings'] = settings
        else:
            self.input['settings'].update(settings)

        super(Gaussian, self).calc_minimize(
            electronic_steps=electronic_steps,
            ionic_steps=ionic_steps,
            algorithm=algorithm,
            ionic_force_tolerance=ionic_forces
        )


    def calc_static(self, electronic_steps=None, algorithm=None):
        """
            Function to setup the hamiltonian to perform static SCF DFT runs

            **Arguments**

                algorithm (str): SCF algorithm
                electronic_steps (int): maximum number of electronic steps, which can be used to achieve convergence
        """
        settings = {}
        if electronic_steps is not None:
            if not 'SCF' in settings:
                settings['SCF'] = []
            settings['SCF'].append("MaxCycle={}".format(electronic_steps))

        if algorithm is not None:
            if not 'SCF' in settings:
                settings['SCF'] = []
            settings['SCF'].append(algorithm)

        self.input['jobtype'] = 'sp'
        if not isinstance(self.input['settings'],dict):
            self.input['settings'] = settings
        else:
            self.input['settings'].update(settings)

        super(Gaussian, self).calc_static(
            electronic_steps=electronic_steps,
            algorithm=algorithm
        )


    def calc_md(self, temperature=None,  n_ionic_steps=1000, time_step=None, n_print=100):
        raise NotImplementedError("calc_md() not implemented in Gaussian.")
    

    def calc_scan(self, nsteps, stepsize, ic):
        '''
            Function to setup the hamiltonian to perform a relaxed scan job
            Starting from the initial value of the IC, stepsize is added nstep times
            to construct a grid for the relaxed scan

            **Arguments**

                nsteps (int)
                            number of steps
                stepsize (float)
                            size of step which is performed nstep times
                ic (tuple)
                            see https://gaussian.com/gic/ for a definition of the structural parameters
                            defined as (kind, [i, j, ...]), where kind equals ('Bond', 'B', 'Dihedral', 'D', ...)
                            and [i,j,...] is the corresponding list of indices, starting to count from 0

        '''
        settings = {'geom':['addgic'], 'nosymm':[]}
        suffix = "Scan(Stepsize={},NStep={}) = {}({})".format(float(stepsize),int(nsteps),ic[0],",".join([str(i+1) for i in ic[1]])) # add 1 since gic starts counting from 1

        self.input['jobtype'] = 'opt'

        if not isinstance(self.input['settings'],dict):
            self.input['settings'] = settings
        else:
            self.input['settings'].update(settings)

        if self.input['suffix'] is not None:
            self.input['suffix'] += '\n' + suffix
        else:
            self.input['suffix'] = suffix


    def print_MO(self):
        """
            Print a list of the MO's with the corresponding orbital energy and occupation.
        """

        n_MO = self.get('output/structure/dft/scf_density').shape[0]
        for index in range(n_MO):
            # print orbital information
            occ_alpha = self.get('output/structure/dft/alpha_occupations')[index]
            occ_beta = self.get('output/structure/dft/beta_occupations')[index]

            if self.get('output/structure/dft/beta_orbital_e') is None:
                orbital_energy = self.get('output/structure/dft/alpha_orbital_e')[index]
                print("#{}: \t Orbital energy = {:>10.5f} \t Occ. = {}".format(index, orbital_energy, occ_alpha + occ_beta))
            else:
                orbital_energy = [self.get('output/structure/dft/alpha_orbital_e')[index], self.get('output/structure/dft/beta_orbital_e')[index]]
                print("#{}: \t Orbital energies (alpha,beta) = {:>10.5f},{:>10.5f} \t Occ. = {},{}".format(index, orbital_energy[0], orbital_energy[1], occ_alpha,occ_beta))


    def visualize_MO(self, index, particle_size=0.5, show_bonds=True):
        """
            Visualize the MO identified by its index.

            **Arguments**

            index       index of the MO, as listed by print_MO()

            particle_size
                        size of the atoms for visualization, lower value if orbital is too small to see

            show_bonds  connect atoms or not

            **Notes**

            This function should always be accompanied with the following commands (in a separate cell)

            view[1].update_surface(isolevel=1, color='blue', opacity=.3)
            view[2].update_surface(isolevel=-1, color='red', opacity=.3)

            This makes sure that the bonding and non-bonding MO's are plotted and makes them transparent
        """
        n_MO = self.get('output/structure/dft/scf_density').shape[0]
        assert index >= 0 and index < n_MO
        assert len(self.get('output/structure/numbers')) < 50 # check whether structure does not become too large for interactive calculation of cube file

        if self.get('output/structure/dft/beta_orbital_e') is None:
            occ = self.get('output/structure/dft/occupations')[index]
            orbital_energy = self.get('output/structure/dft/alpha_orbital_e')[index]
            print("Orbital energy = {:>10.5f} \t Occ. = {}".format(orbital_energy, occ))
        else:
            occ_alpha = self.get('output/structure/dft/alpha_occupations')[index]
            occ_beta = self.get('output/structure/dft/beta_occupations')[index]
            orbital_energy = [self.get('output/structure/dft/alpha_orbital_e')[index], self.get('output/structure/dft/beta_orbital_e')[index]]
            print("Orbital energies (alpha,beta) = {:>10.5f},{:>10.5f} \t Occ. = {},{}".format(orbital_energy[0], orbital_energy[1], occ_alpha, occ_beta))

        # make cube file
        path = self.path+'_hdf5/'+self.name+'/input'
        out = subprocess.check_output(
                "ml load Gaussian/g16_C.02-NVHPC-24.9; cubegen 1 MO={} {}.fchk {}.cube".format(index+1, path,path), # this is at the moment HPC-specific
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True,
            )
        # visualize cube file
        try:
            import nglview
        except ImportError:
            raise ImportError("The visualize_MO() function requires the package nglview to be installed")

        atom_numbers = []
        atom_positions = []

        with open('{}.cube'.format(path),'r') as f:
            for i in range(2):
                f.readline()
            n_atoms = int(f.readline().split()[0][1:])
            for i in range(3):
                f.readline()
            for n in range(n_atoms):
                line = f.readline().split()
                atom_numbers.append(int(line[0]))
                atom_positions.append(np.array([float(m) for m in line[2:]]) * Bohr) 

        structure = Atoms(numbers=np.array(atom_numbers), positions=atom_positions)
        view = nglview.show_ase(structure)
        if not show_bonds:
            view.add_spacefill(radius_type='vdw', scale=0.5, radius=particle_size)
            view.remove_ball_and_stick()
        else:
            view.add_ball_and_stick()
        view.add_component('{}.cube'.format(path))
        view.add_component('{}.cube'.format(path))
        return view


    def read_NMA(self):
        """
            Reads the NMA output from the Gaussian .log file.

            Returns:
                    IR frequencies, intensities and corresponding eigenvectors (modes).
        """
        assert self.input['jobtype'] == 'freq' or self.input['jobtype'] == 'freq(noraman)', "Normal modes available only in a frequency job!" 
        # Read number of atoms
        nrat = len(self.get('output/structure/numbers'))

        # Read IR frequencies and intensities from log file
        freqs = []
        ints = []
        modes = [[] for i in range(nrat)]

        path = self.path+'_hdf5/'+self.name+'/input.log'
        with open(path,'r') as f:
            lines = f.readlines()

        # Assert normal termination
        assert "Normal termination of Gaussian" in lines[-1]

        # Find frequencies and intensities
        for n in range(len(lines)):
            line = lines[n]
            if 'Frequencies --' in line: 
                freqs += [float(i) for i in line[20:].split()]
            elif 'IR Inten    --' in line:
                ints += [float(i) for i in line[15:].split()]
            elif 'Atom  AN      X      Y      Z' in line:
                for m in range(nrat):
                    modes[m] += [float(i) for i in lines[n+m+1][10:].split()]
            else:
                continue

        assert len(freqs) == len(ints), "Number of frequencies and intensities do not match!"

        freqs = np.array(freqs)
        ints = np.array(ints)
        modes = np.array(modes).reshape(len(ints),nrat,3)

        return freqs, ints, modes


    def plot_IR_spectrum(self, width=10, scale=1.0, min_freq=0, max_freq=5000):
        """
            Plots the IR spectrum based on the Gaussian output.
            The peaks in the spectrum are arbitrarily based on Lorentzian widths.

            **Arguments**
            
            width (float): width of the Lorentzian peaks

            scale (float): scaling factor for the frequencies

            min_freq (float): minimum frequency to plot (inverse cm)

            max_freq (float): maximum frequency to plot (inverse cm)
        """
        assert self.input['jobtype'] == 'freq' or self.input['jobtype'] == 'freq(noraman)', "Normal modes available only in a frequency job!" 
        freqs, ints, modes = self.read_NMA()
        xr = np.arange(min_freq, max_freq, 1)
        alphas = np.zeros(len(xr))

        freqs = np.array(freqs) * scale
        for n, (freq, intensity) in enumerate(zip(freqs, ints)):
            alphas += intensity * 1.0 / (1.0 + ((freq - xr) / (width / 2.0))**2) # lorentzian line shape function
            
        fig, ax = pt.subplots()
        ax.plot(xr, alphas)
        ax.set_xlabel("Frequency [1/cm]")
        ax.set_ylabel("Absorption [a.u.]")
        fig.show()


    def animate_nma_mode(self, index, amplitude=1.0, frames=24, spacefill=False, particle_size=0.5):
        """
            Visualize the normal mode corresponding to an index

            **Arguments**

            index: index corresponding to a normal mode

            amplitude: size of the deviation of the normal mode

            frames: number of frames that constitute the full mode (lower means faster movement)

            spacefill: remove atom bonds

            particle size: size of the atoms in the structure
        """
        assert self.input['jobtype'] == 'freq' or self.input['jobtype'] == 'freq(noraman)', "Normal modes available only in a frequency job!" 
        freqs, ints, modes = self.read_NMA()
        print("This mode corresponds to a frequency of {} 1/cm".format(freqs[index]))
        structure = self.get_structure()

        mode = modes[:, index]
        if structure.get_masses() is not None:
            mode /= np.sqrt(structure.get_masses())
        mode /= np.linalg.norm(mode)

        traj = []
        for frame in range(frames):
            _structure = structure.copy()
            factor = amplitude * np.sin(2 * np.pi * float(frame) / frames)
            _structure.set_positions(structure.get_positions() + factor * mode.reshape((-1, 3)))
            traj.append(_structure)

        try:
            import nglview
        except ImportError:
            raise ImportError("The animate_nma_mode() function requires the package nglview to be installed")

        animation = nglview.show_asetraj(traj)
        if spacefill:
            animation.add_spacefill(radius_type='vdw', scale=0.5, radius=particle_size)
            animation.remove_ball_and_stick()
        else:
            animation.add_ball_and_stick()
        return animation


    def bsse_to_pandas(self):
        """
        Convert bsse output of all frames to a pandas Dataframe object.

        Returns:
            pandas.Dataframe: output as dataframe
        """
        assert 'counterpoise' in [k.lower() for k in self.input['settings'].keys()] # check if there was a bsse calculation
        tmp = {}
        with self.project_hdf5.open('output/structure/bsse') as hdf:
            for key in hdf.list_nodes():
                tmp[key] = hdf[key] if isinstance(hdf[key],np.ndarray) else [hdf[key]]
            df = pandas.DataFrame(tmp)
        return df
    

    def animate_scan(self, index=None, spacefill=True, stride=1, center_of_mass=False, particle_size=0.5, plot_energy=False):
        '''
        Animates a scan job. If index is None (default), a trajectory is created using the optimal geometry for
        every frame of the scan. If index is not None, a trajectory is created for that specific frame index,
        showing the geometry optimization at that scan frame.

        Set plot_energy=True if you want the corresponding energy plot.

        Args:
            index (int): index of the frame in the scan trajectory
            spacefill (bool):
            stride (int): show animation every stride [::stride]
                          use value >1 to make animation faster
                           default=1
            center_of_mass (list/numpy.ndarray): The center of mass

        Returns:
            animation: nglview IPython widget

        '''
        assert self.get('output/jobtype')=='scan'

        if index is None:
            energies = self.get('output/generic/energy_tot')
            positions = self.get('output/generic/positions')
        else:
            energies = self.get('output/structure/scan/energies/p{}'.format(index))
            positions = self.get('output/structure/scan/positions/p{}'.format(index))

        if plot_energy:
            pt.clf()
            pt.plot(energies,'bo--')
            if index is None:
                pt.xlabel('Frame')
            else:
                pt.xlabel('Opt steps')
            pt.ylabel('Total energy [a.u.]')
            pt.show()


        # Create the trajectory object
        indices = self.output.indices
        max_pos = np.max(np.max(positions, axis=0), axis=0)
        max_pos[np.abs(max_pos) < 1e-2] = 10
        cell = np.eye(3) * max_pos
        cells = np.array([cell] * len(positions))

        if len(positions) != len(cells):
            raise ValueError("The positions must have the same length as the cells!")

        trajectory = Trajectory(
            positions[::stride],
            self.structure.get_parent_basis(),
            center_of_mass=center_of_mass,
            cells=cells[::stride],
            indices=indices[::stride],
        )

        try:
            import nglview
        except ImportError:
            raise ImportError(
                "The animate() function requires the package nglview to be installed"
            )

        animation = nglview.show_asetraj(trajectory)
        if spacefill:
            animation.add_spacefill(radius_type="vdw", scale=0.5, radius=particle_size)
            animation.remove_ball_and_stick()
        else:
            animation.add_ball_and_stick()
        return animation


    def get_scan_structure(self, index, step=-1):
        '''
        Return the structure from scan 'index' at optimization step 'frame'

        Args:
            index (int): index of the frame in the scan trajectory
            step (int): index of the optimization step in this frame
        '''

        structure = self.structure.copy()
        structure.positions = self.get('output/structure/scan/positions/p{}'.format(index))[step]
        return structure    


class GaussianInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(GaussianInput, self).__init__(input_file_name=input_file_name, table_name="input_inp", comment_char="#")

    def load_default(self):
        '''
        Loading the default settings for the input file.
        '''
        input_str = """\
lot HF
basis_set 6-311G(d,p)
spin_mult 1
charge 0
"""
        self.load_string(input_str)


def write_input(input_dict,working_directory='.'):
    # Comments can be written with ! in Gaussian
    # Load dictionary
    lot          = input_dict['lot']
    basis_set    = input_dict['basis_set']
    spin_mult    = input_dict['spin_mult'] # 2S+1
    charge       = input_dict['charge']
    symbols      = input_dict['symbols']
    pos          = input_dict['pos']
    assert pos.shape[0] == len(symbols)

    # Optional elements
    if not input_dict['mem'] is None:
        mem = input_dict['mem'] + 'B' * (input_dict['mem'][-1]!='B') # check if string ends in bytes
        # convert pmem to mem
        cores = input_dict['cores']
        nmem = str((int(re.findall("\d+", mem)[0]) - 100) * cores) # shave off 100MB per core since Gaussian overextends its memory usage
        mem_unit = re.findall("[a-zA-Z]+", mem)[0]
        mem = nmem+mem_unit
    else:
        mem = "800MB" # default allocation

    if not input_dict['jobtype'] is None:
        jobtype = input_dict['jobtype']
        if " " in jobtype:
            warnings.warn('Please refrain from specifying multiple jobtypes or settings in the jobtype. To specify different settings, use the settings dictionary instead.')
    else:
        jobtype = "" # corresponds to sp

    if not input_dict['title'] is None:
        title = input_dict['title']
    else:
        title = "no title"

    if not input_dict['settings'] is None:
        settings = input_dict['settings'] # dictionary {key: [options]}
    else:
        settings = {}

    if input_dict['suffix'] is not None:
        input_dict['suffix'] = input_dict['suffix'].strip() # remove leading and trailing whitespaces

    verbosity_dict={'low':'t','normal':'n','high':'p'}
    if not input_dict['verbosity'] is None:
        verbosity  = input_dict['verbosity']
        if verbosity in verbosity_dict:
            verbosity = verbosity_dict[verbosity]
    else:
        verbosity='n'

    settings_keys = [key.lower() for key in settings.keys()]
    if 'counterpoise' in settings_keys:
        if input_dict['bsse_idx'] is None or not len(input_dict['bsse_idx'])==len(pos) : # check if all elements are present for a BSSE calculation
            raise ValueError('The Counterpoise setting requires a valid bsse_idx array')
        # Check bsse idx (should start from 1 for Gaussian)
        input_dict['bsse_idx'] = [k - min(input_dict['bsse_idx']) + 1 for k in input_dict['bsse_idx']]
        # Check if it only contains consecutive numbers (sum of set should be n*(n+1)/2)
        assert sum(set(input_dict['bsse_idx'])) == (max(input_dict['bsse_idx'])*(max(input_dict['bsse_idx']) + 1))/2

    if 'empiricaldispersion' in settings_keys:
        if verbosity in ['t','n']:
            warnings.warn('You can only use the EmpiricalDispersion option with a "high" verbosity. This has been automatically updated.')
            verbosity = 'p'

    if 'geom' in settings_keys and 'addgic' in settings['geom']:
        assert input_dict['suffix'] is not None

    # Parse settings
    settings_string = ""
    for key,valuelst in settings.items():
        if not isinstance(valuelst, list):
            valuelst = [valuelst]
        option = key + "({})".format(",".join([str(v) for v in valuelst]))*(len(valuelst)>0) + ' '
        settings_string += option

    # Spin-orbit check
    lot_line = "".join(lot.lower().split())
    if 'spin' in lot_line and not 'nroot' in lot_line:
        lot_line = lot_line[:-1] + ',nroot=1)' # add the nroot option if it is not given

    # If CAS calculation create suffix
    if 'cas' in lot_line:
        if input_dict['suffix'] is None:
            input_dict['suffix'] = ""
        else:
            input_dict['suffix']+= '\n\n' # separation between provided suffix and cas suffix

        if 'nroot' in lot_line:
            nroot = int(re.search(r'nroot=\s*(\d+)', lot_line).group(1))
            weights = [np.round(1./nroot,2) for i in range(nroot)]
            if not sum(weights) == 1:
                weights[-1] = np.round(1-sum(weights[:-1]),2)

            input_dict['suffix'] += " ".join([str(w) for w in weights])
            input_dict['suffix'] += '\n\n'

        if 'spin' in lot_line:
            assert input_dict['spin_orbit_states'] is not None
            assert len(input_dict['spin_orbit_states']) == 2
            assert all(isinstance(state, int) for state in input_dict['spin_orbit_states'])

            input_dict['suffix'] += " ".join([str(state) for state in input_dict['spin_orbit_states']])
            input_dict['suffix'] += '\n\n'

    # Write to file
    route_section = "#{} {}/{} {} {}\n\n".format(verbosity,lot_line,basis_set,jobtype,settings_string)
    with open(os.path.join(working_directory, 'input.com'), 'w') as f:
        f.write("%mem={}\n".format(mem))
        f.write("%chk=input.chk\n")
        f.write(route_section)
        f.write("{}\n\n".format(title))

        if not 'Counterpoise' in settings.keys():
            f.write("{} {}\n".format(charge,spin_mult))
            for n,p in enumerate(pos):
                f.write(" {}\t{: 1.6f}\t{: 1.6f}\t{: 1.6f}\n".format(symbols[n],p[0],p[1],p[2]))
            f.write('\n')
        else:
            if isinstance(charge,list) and isinstance(spin_mult,list): # for BSSE it is possible to define charge and multiplicity for the fragments separately
                f.write(" ".join(["{},{}".format(charge[idx],spin_mult[idx]) for idx in range(int(settings['Counterpoise']))])) # first couple is for full system, then every fragment separately
            else:
                f.write("{} {}\n".format(charge,spin_mult))

            for n,p in enumerate(pos):
                f.write(" {}(Fragment={})\t{: 1.6f}\t{: 1.6f}\t{: 1.6f}\n".format(symbols[n],input_dict['bsse_idx'][n],p[0],p[1],p[2]))
            f.write('\n')

        if input_dict['suffix'] is not None:
            f.write(input_dict['suffix'])
        f.write('\n\n')



def fchk2dict(output_file):
    # probably still some data missing
    # check job type, for now implement basics (energy=single point, opt = full opt, freq = frequency calculation)
    fchk = load_one(output_file, fmt='fchk')
    if not fchk.run_type in ['energy','opt','freq']:
        raise NotImplementedError

    # Basic information
    fchkdict = {}
    if fchk.run_type == 'energy':
        fchkdict['jobtype']     = 'sp'
    else:
        fchkdict['jobtype']     = fchk.run_type
    fchkdict['lot']         = fchk.lot
    fchkdict['basis_set']   = fchk.obasis_name

    fchkdict['structure/numbers']     = fchk.atnums
    fchkdict['structure/masses']      = fchk.atmasses * _me / _amu # from atomic units to amu
    fchkdict['structure/charges']     = fchk.atcharges 
    fchkdict['structure/dipole']      = None
    for key, item in fchk.moments.items():
        if key == (1, 'c'): # dipole moment in cartesian coordinates
            fchkdict['structure/dipole'] = item
    fchkdict['structure/dft/n_electrons']         = fchk.nelec
    fchkdict['structure/dft/n_basis_functions']   = len(fchk.mo.coeffs)
    fchkdict['structure/dft/occupations']         = fchk.mo.occs
    fchkdict['structure/dft/alpha_occupations']   = fchk.mo.occsa
    fchkdict['structure/dft/beta_occupations']    = fchk.mo.occsb

    # Orbital information
    if fchk.mo.kind == 'restricted':
        fchkdict['structure/dft/alpha_orbital_e'] = fchk.mo.energies
        fchkdict['structure/dft/beta_orbital_e']  = None
    elif fchk.mo.kind == 'unrestricted':
        fchkdict['structure/dft/alpha_orbital_e'] = fchk.mo.energies[:fchk.mo.norba]
        fchkdict['structure/dft/beta_orbital_e']  = fchk.mo.energies[fchk.mo.norba:]

    # Densities
    fchkdict['structure/dft/scf_density'] = fchk.one_rdms.get("scf", None)
    fchkdict['structure/dft/spin_scf_density'] = fchk.one_rdms.get("scf_spin", None)
    fchkdict['structure/dft/post_scf_density'] = fchk.one_rdms.get("post_scf_ao", None)
    fchkdict['structure/dft/post_spin_scf_density'] = fchk.one_rdms.get("post_scf_spin_ao", None)

    def _generate_indices(atnums):
        # indices indicates which atom belongs to which element (e.g. ['O', 'H', 'H'] -> [0, 1, 1])
        mapping = {}
        indices = []
        for atnum in atnums:
            if atnum not in mapping:
                mapping[atnum] = len(mapping)
            indices.append(mapping[atnum])
        return indices

    fchkdict['structure/positions']   = [fchk.atcoords * Bohr] # from a.u. to A
    # Specific job information
    if fchkdict['jobtype'] == 'opt':
        indices, cells, positions, energy_tot, forces = [], [], [], [], []
        for f in load_many(output_file, fmt='fchk'):
            indices.append(_generate_indices(f.atnums)) # needed to get structure in ase format, an error is encountered otherwise
            cells.append(None) # needed to get structure in ase format, an error is encountered otherwise
            positions.append(f.atcoords * Bohr) # from a.u. to A
            energy_tot.append(f.energy * Ha) # from a.u. to eV
            forces.append(f.atgradient * -1 * Ha / Bohr) # from a.u. to eV/A

        fchkdict['generic/indices']       = indices
        fchkdict['generic/cells']         = cells
        fchkdict['generic/positions']     = np.array(positions) 
        fchkdict['generic/energy_tot']    = energy_tot
        fchkdict['generic/forces']        = np.array(forces)

    if fchkdict['jobtype'] == 'freq':
        fchkdict['generic/indices']       = [_generate_indices(fchk.atnums)] # needed to get structure in ase format, an error is encountered otherwise
        fchkdict['generic/cells']         = [None] # needed to get structure in ase format, an error is encountered otherwise
        fchkdict['generic/positions']     = np.array([fchk.atcoords * Bohr]) # from a.u. to A
        fchkdict['generic/forces']        = np.array([fchk.atgradient * -1 * Ha / Bohr]) # from a.u. to eV/A
        fchkdict['generic/hessian']       = [fchk.athessian * Ha / (Bohr**2)] # from a.u. to eV/A^2
        fchkdict['generic/energy_tot']    = [fchk.energy * Ha] # from a.u. to eV

    if fchkdict['jobtype'] == 'sp':
        fchkdict['generic/indices']       = [_generate_indices(fchk.atnums)] # needed to get structure in ase format, an error is encountered otherwise
        fchkdict['generic/cells']         = [None] # needed to get structure in ase format, an error is encountered otherwise
        fchkdict['generic/positions']     = np.array([fchk.atcoords * Bohr]) # from a.u. to A
        fchkdict['generic/energy_tot']    = [fchk.energy * Ha] # from a.u. to eV

    return fchkdict


def get_bsse_array(line,it):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    cE_corr = float(rx.findall(line)[0]) * kcalmol/electronvolt
    line = next(it) # go to next line
    cE_raw = float(rx.findall(line)[0]) * kcalmol/electronvolt
    line = next(it) # go to next line
    sum_fragments = float(rx.findall(line)[0])/electronvolt
    line = next(it) # go to next line
    bsse_corr = float(rx.findall(line)[0])/electronvolt
    line = next(it) # go to next line
    E_tot_corr = float(rx.findall(line)[0])/electronvolt

    return E_tot_corr,bsse_corr,sum_fragments,cE_raw,cE_corr


def read_bsse(output_file,output_dict):
    # Check whether the route section contains the Counterpoise setting (if fchk module is update, route section can be loaded from dict)
    cp = False
    with open(output_file,'r') as f:
        line = f.readline()
        while line:
            if 'route' in line.lower():
                if 'counterpoise' in f.readline().lower(): # read next line
                    cp = True
                break
            line = f.readline()

    if cp:
        # the log file has the same path and name as the output file aside from the file extension
        log_file = output_file[:output_file.rfind('.')] + '.log'

        frames = 1 if isinstance(output_dict['generic/energy_tot'],float) else len(output_dict['generic/energy_tot'])

        output_dict['structure/bsse/energy_tot_corrected'] = np.zeros(frames)
        output_dict['structure/bsse/bsse_correction'] = np.zeros(frames)
        output_dict['structure/bsse/sum_of_fragments'] = np.zeros(frames)
        output_dict['structure/bsse/complexation_energy_raw'] = np.zeros(frames)
        output_dict['structure/bsse/complexation_energy_corrected'] = np.zeros(frames)

        it = _reverse_readline(log_file)
        line = next(it)
        for i in range(frames):
            found = False
            while not found:
                line = next(it)
                if 'complexation energy' in line:
                    E_tot_corr,bsse_corr,sum_fragments,cE_raw,cE_corr = get_bsse_array(line,it)
                    output_dict['structure/bsse/energy_tot_corrected'][i] = E_tot_corr
                    output_dict['structure/bsse/bsse_correction'][i] = bsse_corr
                    output_dict['structure/bsse/sum_of_fragments'][i] = sum_fragments
                    output_dict['structure/bsse/complexation_energy_raw'][i] = cE_raw
                    output_dict['structure/bsse/complexation_energy_corrected'][i] = cE_corr
                    found = True

        if frames==1:
            output_dict['structure/bsse/energy_tot_corrected'] = output_dict['structure/bsse/energy_tot_corrected'][0]
            output_dict['structure/bsse/bsse_correction'] = output_dict['structure/bsse/bsse_correction'][0]
            output_dict['structure/bsse/sum_of_fragments'] = output_dict['structure/bsse/sum_of_fragments'][0]
            output_dict['structure/bsse/complexation_energy_raw'] = output_dict['structure/bsse/complexation_energy_raw'][0]
            output_dict['structure/bsse/complexation_energy_corrected'] = output_dict['structure/bsse/complexation_energy_corrected'][0]
        else:
            # flip array sequence
            output_dict['structure/bsse/energy_tot_corrected'] = output_dict['structure/bsse/energy_tot_corrected'][::-1]
            output_dict['structure/bsse/bsse_correction'] = output_dict['structure/bsse/bsse_correction'][::-1]
            output_dict['structure/bsse/sum_of_fragments'] = output_dict['structure/bsse/sum_of_fragments'][::-1]
            output_dict['structure/bsse/complexation_energy_raw'] = output_dict['structure/bsse/complexation_energy_raw'][::-1]
            output_dict['structure/bsse/complexation_energy_corrected'] = output_dict['structure/bsse/complexation_energy_corrected'][::-1]


def read_EmpiricalDispersion(output_file,output_dict):
    # Get dispersion term from log file if it is there
    # dispersion term is not retrieved from gaussian output in fchk

    disp = None
    with open(output_file,'r') as f:
        while True:
            line = f.readline()
            if 'Route' in line:
                line = f.readline()
                if 'EmpiricalDispersion' in line:
                    idx = line.find('EmpiricalDispersion')
                    if 'GD3' in line[idx:]:
                        search_term = 'Grimme-D3 Dispersion energy='
                    else:
                        raise NotImplementedError
                else:
                    return
                break

    # the log file has the same path and name as the output file aside from the file extension
    log_file = output_file[:output_file.rfind('.')] + '.log'
    it = _reverse_readline(log_file)
    while True:
        line = next(it)
        if search_term in line:
            disp = float(line[38:-9])/electronvolt # could be changed when new search terms are implemented
            break

    output_dict['generic/energy_tot'] += disp


def _collect_output(output_file):
    # Translate to dict
    output_dict = fchk2dict(output_file)

    # Read BSSE output if it is present
    read_bsse(output_file, output_dict)

    # Correct energy if empirical dispersion contribution is present
    read_EmpiricalDispersion(output_file, output_dict)

    return output_dict


def _reverse_readline(filename, buf_size=8192):
    """
    A generator that returns the lines of a file in reverse order

    https://stackoverflow.com/questions/2301789/read-a-file-in-reverse-order-using-python
    """
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment
