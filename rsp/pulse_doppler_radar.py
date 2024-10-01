from .constants import C

def range_unambiguous(PRF):
    """maximum unambigious range"""
    return C/(2*PRF)

def frequency_doppler(rangeRate, f0):
    """frequnce of light recieved after reflection off target with the given rangeRate"""
    return f0*(-2*rangeRate/C)

def frequency_aliased(freq, freq_sample):
    """Place freq in [-freq_sample/2, freq_sample/2]
    Usefull for finding aliasing of real signals"""
    f = freq%freq_sample
    if f > freq_sample/2:
        return  f - freq_sample
    else:
        return f

def rangeRate_pm_unambiguous(PRF, f0):
    "+/- bounds of the unambgeous velocity"
    return PRF*C/(4*f0)

def rangeRate_aliased_rrmax(rangeRate, rangeRate_max):
    """Place freq in [-rangeRate_max, rangeRate_max]
    Usefull for finding aliasing of real signals"""
    r = rangeRate%(2*rangeRate_max)
    if r > rangeRate_max:
        return  r - 2*rangeRate_max
    else:
        return r

def rangeRate_aliased_prf_f0(rangeRate, PRF, f0):
    """Place freq in [-rangeRate_max, rangeRate_max]
    Usefull for finding aliasing of real signals"""
    return rangeRate_aliased_rrmax(rangeRate, rangeRate_pm_unambiguous(PRF, f0))
