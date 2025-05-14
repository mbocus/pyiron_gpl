import numpy as np
import matplotlib.pyplot as plt

from ase.units import kB
from scipy.stats import chi2


def get_slice(job, start=0, end=-1, max_sample=None, step=None):
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

    # make an array with the atomic weights used to compute the temperature
    structure = job.get_structure(-1)
    weights = np.array(structure.get_masses()) / kB
    
    # load optional arguments
    start, end, step = get_slice(job, **kwargs)

    # load temperatures from output
    temps = job.content["output/generic/temperature"][start:end:step]
    if temp is None:
        temp = temps.mean()

    # system definition
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
    rv = chi2(ndof, 0, temp / ndof)
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