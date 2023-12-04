"""
Collection of utility functions.

"""

import itertools

from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen


__date__ = "2020-07-23"
__version__ = "0.1"


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    """
    xs = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(xs, n) for n in range(len(xs)+1))


def collect_entries_vasp(datapath, ncores=1, outfile="entries.json",
                         inc_structure=False, parameters=None,
                         data=None, verbose=False):
    """
    Get entries (i.e., compositions and energies) from local VASP
    calculations by searching through a directory tree.

    Arguments:
      datapath (str)    One or more root path(s) for data assimilation
      ncores (int)      Number of cores to use
      outfile (str)     File for saving collected entries
      inc_structure     If True, include atomic structures in entry
      parameters (list) Input file parameters to keep track of.
      data (list)       Additional output data that should be retrieved
                        from the VASP calculations.  Per default, the
                        'filename' is kept track of.
      verbose (bool)    If true, write number of found entries to stdout

    """
    # Always include the file name
    if data is None:
        data = ['filename']
    else:
        data.append('filename')

    drone = VaspToComputedEntryDrone(inc_structure=inc_structure,
                                     parameters=parameters, data=data)

    if isinstance(datapath, (list, tuple)):
        local_entries = []
        for path in datapath:
            queen = BorgQueen(drone, rootpath=path,
                              number_of_drones=ncores)
            local_entries += queen.get_data()
        queen._data = local_entries
    else:
        queen = BorgQueen(drone, rootpath=datapath,
                          number_of_drones=ncores)
        local_entries = queen.get_data()
    queen.save_data(outfile)

    # filter out invalid entries
    real_local_entries = []
    for entry in local_entries:
        if entry is not None:
            real_local_entries.append(entry)

    if verbose:
        print("{} local entries found.".format(len(real_local_entries)))
        print("Saving to file: {}".format(outfile))

    return real_local_entries


def load_entries_json(entryfiles, verbose=False):
    """
    Load VASP calculation entries from a json file.

    Arguments:
      entryfiles (str)  Path to a JSON file containing entries,
                        or list of paths to several such files
      verbose (bool)    If true, print additional info

    """

    drone = VaspToComputedEntryDrone()

    queen = BorgQueen(drone)
    if isinstance(entryfiles, (list, tuple)):
        local_entries = []
        for fname in entryfiles:
            queen.load_data(fname)
            local_entries += queen.get_data()
        queen._data = local_entries
    else:
        queen.load_data(entryfiles)
        local_entries = queen.get_data()

    # filter out invalid entries
    real_local_entries = []
    for entry in local_entries:
        if entry is not None:
            real_local_entries.append(entry)

    if verbose:
        print("{} entries loaded from file(s): {}".format(
            len(real_local_entries), entryfiles))

    return real_local_entries
