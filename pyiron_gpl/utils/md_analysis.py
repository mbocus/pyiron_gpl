import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2
from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.units import fs, invcm


def _get_slice(job, start=0, end=-1, max_sample=None, step=None):
    dimensions = job.content["output/generic/positions"].shape
    nrow = dimensions[0] if len(dimensions) == 3 else 1

    if end < 0:
        end = nrow + end + 1
    else:
        end = min(end, nrow)

    if start < 0:
        start = nrow + start + 1

    if step is None:
        if max_sample is None:
            return start, end, 1
        else:
            assert end > 0
            step = max(1, (end - start) // max_sample + 1)

    return start, end, step


def _iter_indices(array, indices):
    if indices is None:
        for idx in np.npdindex(array.shape[1:]):
            yield idx
    else:
        for idx in indices:
            for idx_rest in np.ndindex(array.shape[2:]):
                yield (idx,) + idx_rest


def plot_temp_dist(job, temp=None, ndof=None, **kwargs):
    """
    Plot the distribution of the weighted atomic velocities.

    **Arguments**
    job
        A pyiron job containing the MD data.
    
    **Optional arguments**
    temp
        The (expected) average temperature used to plot the theoretical distributions.
    ndof
        The number of degrees of freedom. If not specified, it is by default 3*(number of atoms)

    **Optional**
    start
        The first sample to be considered for analysis. This may be negative to indicate that the analysis should start from the -start last sample.
    end
        The last sample to be considered. This may be negative to indicate that the last -end samples should not be considered.
    max_sample
        When give, step is set such that the number of samples does not exceed max_sample.
    step
        The spacing between the samples used for the analysis.

    This type of plot is useful to check the sanity of a simulation.
    The empirical cumulative distribution is plotted and overlayed with the analytical cumulative distribution one would expect from the NVT ensemble.
    This type of plot helps in identifying the lack of thermal equilibrium.
    """
    
    # load optional arguments
    start, end, step = _get_slice(job, **kwargs)

    # load temperatures from output
    temps = job.content["output/generic/temperature"][start:end:step]
    if temp is None:
        temp = temps.mean()

    # system definition
    structure = job.get_structure(-1)
    natom = structure.get_atomic_numbers().shape[0]
    if ndof is None:
        ndof = 3 * natom
    sigma = temp * np.sqrt(2.0 / ndof)
    temp_step = sigma / 5

    # setup the temperature grid and make the histogram
    temp_grid = np.arange(max(0, temp - 3*sigma), temp + 5 * sigma, temp_step)
    counts = np.histogram(temps.ravel(), bins=temp_grid)[0]
    total = float(len(temps))

    # transform into empirical pdf and cdf
    emp_sys_pdf = counts / total
    emp_sys_cdf = counts.cumsum() / total

    # analytical form
    rv = chi2(df=ndof, loc=0, scale=temp/ndof)
    x_sys = temp_grid[:-1]
    ana_sys_pdf = rv.cdf(temp_grid[1:]) - rv.cdf(temp_grid[:-1])
    ana_sys_cdf = rv.cdf(temp_grid[1:])

    # plot results
    fig, axs = plt.subplots(2, 1, sharex=True)
    scale = 1 / emp_sys_pdf.max()

    axs[0].plot(x_sys, emp_sys_pdf * scale, "k-", drawstyle="steps-pre", label="Simulation ({:.0f} K)".format(temps.mean()))
    axs[0].plot(x_sys, ana_sys_pdf * scale, "r-", drawstyle="steps-pre", label="Exact ({:.0f} K)".format(temp))
    axs[0].axvline(temp, color="k", ls="--")
    axs[0].set_ylim(ymin=0)
    axs[0].set_xlim(x_sys[0], x_sys[-1])
    axs[0].set_ylabel("Rescaled PDF")
    axs[0].legend(loc=0)

    axs[1].plot(x_sys, emp_sys_cdf, "k-", drawstyle="steps-pre")
    axs[1].plot(x_sys, ana_sys_cdf, "r-", drawstyle="steps-pre")
    axs[1].axvline(temp, color="k", ls="--")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(x_sys[0], x_sys[-1])
    axs[1].set_ylabel("CDF")
    axs[1].set_xlabel("Temperature [K]")

    fig.tight_layout()


def plot_rdf(job, rmax=None, width=None, elements=None, start=0, end=-1, step=1):
    """
    Plot the radial distribution function of a set of atoms.

    **Arguments**
    job
        A pyiron job.
    
    **Optional arguments**
    rmax
        The cutoff radius to compute the RDF, in angstrom (if None => half of the shortest unit cell vector of the last frame)
    width
        The resolution of the RDF bins (if None => rmax / 100)
    elements
        The elements among which the RDF should be computed.  If elements is an integer or a list/tuple of integers, only those atoms will contribute to the RDF (like a mask). If elements is a string or a list/tuple of strings, only Atoms of those elements will contribute.
    start, end, step
        Compute the RDF slicing the trajectory according to slice(start, stop, step)
    """

    positions = job.content["output/generic/positions"][start:end:step]
    cells = job.content["output/generic/cells"][start:end:step]
    species = job.content["output/structure/species"]
    indices = job.content["output/structure/indices"]
    chemical_symbols = [species[i] for i in indices]

    # convert the pyiron job output attributes to a list of atoms
    traj = []
    for i, position in enumerate(positions):
        atoms = Atoms(symbols=chemical_symbols, positions=position, cell=cells[i], pbc=True)
        traj.append(atoms)
    
    if rmax is None:
        rmax = np.min(traj[-1].get_cell_lengths_and_angles()[:3]) / 2
    if width is None:
        width = rmax / 100

    nbins = rmax // width
    analysis = Analysis(traj)
    rdf, distances = analysis.get_rdf(rmax, nbins, elements=elements, return_dists=True)

    # compute coordination number
    coordination_number = np.cumsum(rdf * (4 * np.pi * distances**2 * width))

    # plot the rdf
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(distances, rdf)
    axs[0].set_ylabel("RDF")

    axs[1].plot(distances, coordination_number)
    axs[1].set_ylabel("Coordination number")
    axs[1].set_xlabel("Distance [Angstrom]")

    fig.tight_layout()


def plot_velocity_spectrum(job, verticals=None, start=0, end=-1, step=1, bsize=4096, indices=None):
    """
        **Arguments**
        job
            job object that contains an MD trajectory of the structure

        **Optional arguments:**
        verticals
            Array containing the thermostat timeconstant of the system at the first index (fs), followed by the wavenumbers (in 1/cm) of the system.
        start
            The first sample to be considered for analysis. This may be negative to indicate that the analysis should start from the -start last samples.
        end
            The last sample to be considered for analysis. This may be negative to indicate that the last -end sample should not be considered.
        step
            The spacing between the samples used for the analysis.
        bsize
            The size of the blocks used for individual FFT calls.
        elements
            A list of atom indexes that are considered for the computation
            of the spectrum. If not given, all atoms are used.

        The max_sample argument from get_slice is not used because the choice step value is an important parameter: it is best to choose step*bsize such that it coincides with a part of the trajectory in which the velocities (or other data) are continuous.

        The block size should be set such that it corresponds to a decent resolution on the frequency axis, i.e. 33356 fs of MD data corresponds to a resolution of about 1 cm^-1. The step size should be set such that the highest frequency is above the highest relevant frequency in the spectrum, e.g. a step of 10 fs corresponds to a frequency maximum of 3336 cm^-1. The total number of FFT's, i.e. length of the simulation divided by the block size multiplied by the number of time-dependent functions in the data, determines the noise reduction on the (the amplitude of) spectrum. If there is sufficient data to perform 10K FFT's, one should get a reasonably smooth spectrum.

        Depending on the FFT implementation in numpy, it may be interesting to tune the bsize argument. A power of 2 is typically a good choice.
    """    
    # read velocities
    velocities = job.content["output/generic/velocities"] * fs # in A/fs

    # read timestep
    input_dict = job.content["input/control_inp/data_dict"]
    steps = job.content["output/generic/steps"]
    timestep = float(input_dict["Value"][input_dict["Parameter"].index("timestep")]) * (steps[1] - steps[0]) # in fs

    ssize = bsize // 2 + 1 # spectrum size
    current = start
    stride = step * bsize
    work = np.zeros(bsize, float)
    freqs = np.arange(ssize) / (timestep * bsize) / fs / invcm # in cm-1
    amplitudes = np.zeros(ssize, float)
    while current <= end - stride:
        for idx in _iter_indices(velocities, indices):
            work = velocities[(slice(current, current+stride, step),) + idx] 
            amplitudes += abs(np.fft.rfft(work))**2
        current += stride

    # plot results
    fig, ax = plt.subplots()
    ax.plot(freqs, amplitudes)
    
    if verticals is not None:
        thermo_freq = 1 / (verticals[0] * fs) / invcm
        #plot frequencies original system, and coupling to thermostat
        for i in np.arange(1, len(verticals)):
            ax.axvline(verticals[i], color='r', ls='--')
            ax.axvline(verticals[i] + thermo_freq, color='g', ls='--')
            ax.axvline(verticals[i] - thermo_freq, color='g', ls='--')

    ax.set_xlim(0, 3500)

    ax.set_xlabel("Wavenumber [1/cm]")
    ax.set_ylabel("Amplitude")