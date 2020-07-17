from gpaw.utilities.watermodel import FixBondLengthsWaterModel as FixBondLengths

def rigid(atoms, qmidx=[], num_cts=0):
    ''' Only works for QM..MM sequences '''
    if len(qmidx) == 0:
        qm = 0
    else:
        qm = qmidx[-1] + 1 + num_cts
    rattle = ([(3 * i + j, 3 * i + (j + 1) % 3)
               for i in range((len(atoms) - qm + 1) // 3)
               for j in [0, 1, 2]])

    rattle = [(c[0] + qm, c[1] + qm) for c in rattle]
    rattle = FixBondLengths(rattle)
    return rattle

