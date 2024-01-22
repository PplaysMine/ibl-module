# REQUIRED LIBRARIES
# - one-api
# - ibllib
# - iblatlas
# - tqdm
# - numpy
# - matplotlib

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.notebook import tqdm

from brainbox.io.spikeglx import Streamer
from iblatlas.regions import BrainRegions

from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, kstest, binomtest

from fooof import FOOOF

import os, pickle
from pathlib import Path
import gc

# Import the ONE (Open Neurophysiology Environment) class from the ONE API package
from one.api import ONE

# Create an instance of the ONE class, specifying the base URL of the server and a password for authentication
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)

# base path for the locally saved data
basePath = 'E:/iblData/data/'


def getProbesForRegion(acronym: str, saveResult: bool = False) -> list:
    """Returns a list of probe IDs the correspond to the given region

    Parameters
    ----------
    acronym: str
        The brain region acronym according to CCFv3
    saveResult: bool, optional
        If the resulting list should be saved (default is False)

    Returns
    -------
    pidsOfRegion: list
        A list of pids
    """
    sessions = one.alyx.rest('insertions', 'list', atlas_acronym=acronym)
    pidsOfRegion = [i['id'] for i in sessions]
    if saveResult:
        np.save(f'{basePath}{acronym}-list.npy', pidsOfRegion)
    return pidsOfRegion

def getRegionsForProbe(pid: str) -> list:
    """Returns a list of acronyms recorded by the probe with the given probe ID

    Parameters
    ----------
    pid: str
        The probe ID of the probe

    Returns
    -------
    acronyms: list
        A list of acronyms
    """
    eid, probe = one.pid2eid(pid)
    sites = set(one.load_object(eid, 'electrodeSites', collection=f"alf/{probe}").brainLocationIds_ccf_2017)
    br = BrainRegions()
    acronyms = [br.id2acronym(x)[0] for x in sites]

    return acronyms

def getMetaDataForPid(pid: str, acronym: str) -> dict:
    """Returns the locally saved metadata for the given recording

    Parameters
    ----------
    pid: str
        The probe ID of the probe
    acronym: str
        The acronmy of the brain region

    Returns
    -------
    meta: dict
        The metadata dictionary

    Raises
    ------
    FileNotFoundError
        If the locally saved metadata file is not found
    """
    metaFile = f'{basePath}{acronym.replace("/", "_")}/{pid}-{acronym.replace("/", "_")}-meta.npy'
    if os.path.isfile(metaFile):
        with open(metaFile, 'rb') as mf:
            meta = pickle.load(mf)
        return meta
    else:
        raise FileNotFoundError(metafile)

def getDataFromPid(pid: str, acronym: str, timeWin: int = 3, saveResult: bool = False, removeInvalidTrials: bool = False, onlyLocal: bool = False) -> tuple:
    """Returns the raw ephys data

    Parameters
    ----------
    pid: str
        The probe ID of the probe
    acronym: str
        The acronmy of the brain region
    timeWin: int, optional
        The duration of the snippet after the go cue. Note that 1 second before the go cue will be added automatically. (default is 3)
    saveResult: bool, optional
        If the mean of the channels should be saved (default is False)
    removeInvalidTrials: bool, optional
        If only valid trials should be considered (default is False)
    onlyLocal: bool, optional
        If only data should be returned that is already saved locally (default is False)

    Returns
    -------
    dataCompressed: list
        The raw LFP data, shape: (# of trials, samplerate * (timeWin + 1))
    times: tuple
        Significant time points. (time of stim. on, time of stim. off, time of feedback, time of first movement, relative time of feedback)
        All times in seconds. stim. on, stim. off & feedback time since start of recording, first movement & relative fb since go cue
    feedbackType: list
        List of feedback type, -1 indicates negative feedback, +1 indicates positive feedback. len(feedbackType) == # of trials
    contrast: list
        List of contrast, values of [0.0625, 0.125, 0.25, 1, 2]. Negative values indicate left side stim, 2 indicates no contrast
    meta: dict
        The metadata dictionary

    Raises
    ------
    Invalid brain region acronym
        If no region could be found for the given brain region acronym
    No insertion
        If no insertions could be found for the given pid
    No data found for brain region
        The session of the given pid & brain region could not be found
    """
    
    br = BrainRegions()
    brainLocationId_ccf_2017 = br.acronym2id(acronym) # Convert brain region acronym to id
    if len(brainLocationId_ccf_2017) == 0: raise Exception("Invalid brain region acronym.")

    dataFile = f'{basePath}{acronym.replace("/", "_")}/{pid}-{acronym.replace("/", "_")}.npy'
    metaFile = f'{basePath}{acronym.replace("/", "_")}/{pid}-{acronym.replace("/", "_")}-meta.npy'

    if (not (os.path.isfile(dataFile) and os.path.isfile(metaFile))) and onlyLocal:
        return np.array([[] for i in range(5)])
    
    eid, probe = one.pid2eid(pid)
    trials = one.load_object(eid, 'trials', collection='alf') # Load trial data (stimOn_times, response_times, etc.)
    probes = one.load_object(eid, 'probes', collection='alf')

    pids = one.eid2pid(eid)[0]

    insertion = one.alyx.rest('insertions', 'list', id=pid, atlas_acronym=acronym)
    if len(insertion) == 0: raise Exception("No insertion")
    allSites = []
    for probe in probes['description']:
        if probe['label'] == insertion[0]['name']:
            try:
                sites = one.load_object(eid, 'electrodeSites', collection=f"alf/{probe['label']}")
                allSites += list(sites.brainLocationIds_ccf_2017)
            except: None

    stimOn_times = trials['stimOn_times']
    stimOff_times = trials['stimOff_times']
    firstMovement_times = trials['firstMovement_times']
    feedback_times = trials['feedback_times']
    feedback_type = trials['feedbackType']

    contrastLeft = trials['contrastLeft']
    contrastRight = trials['contrastRight']

    contrast = np.empty(len(contrastLeft))
    contrast.fill(np.nan)
    for i in range(len(contrast)):
        if np.isnan(contrastLeft[i]):
            if contrastRight[i] == 0: contrast[i] = 2
            else: contrast[i] = contrastRight[i]
        elif np.isnan(contrastRight[i]):
            if contrastLeft[i] == 0: contrast[i] = -2
            else: contrast[i] = -contrastLeft[i]

    channels = [i for i, x in enumerate(allSites) if x == brainLocationId_ccf_2017] # List with indices of channels that are from the specified brain region
    if len(channels) == 0: raise Exception(f'No data found for brain region {acronym} (id: {brainLocationId_ccf_2017})')

    times = (stimOn_times, stimOff_times, feedback_times, np.subtract(firstMovement_times, stimOn_times), np.subtract(feedback_times, stimOn_times))

    if os.path.isfile(dataFile) and os.path.isfile(metaFile):
        print('Files found, loading...')
        with open(metaFile, 'rb') as mf:
            meta = pickle.load(mf)
        with open(dataFile, 'rb') as df:
            data = np.load(df)
        if removeInvalidTrials:
            data, times, contrast, feedback_type = __removeInvalidTrials(data, times, feedback_type, contrast, meta)
        return data, times, feedback_type, contrast, meta
    elif onlyLocal:
        return np.array([[] for i in range(5)])

    sr = Streamer(pid=pid, one=one, remove_cached=False, typ='lf')
    raw = np.empty((len(trials['stimOn_times']), len(channels), round(sr.fs) * (timeWin + 1))) # Create empty raw list

    invalidTrials = []
    for i, t0 in enumerate(tqdm(stimOn_times)):
        if np.isnan(t0): invalidTrials.append(i); continue
        s0 = t0 * round(sr.fs)
        tsel = slice(int(s0) - int(1 * round(sr.fs)), int(s0) + int((timeWin) * round(sr.fs)))
        bufferedArray = [x for i, x in enumerate(sr[tsel, :-sr.nsync].T) if i in channels]
        for j, arr in enumerate(bufferedArray):
            while len(bufferedArray[j]) < round(sr.fs) * (timeWin+1):
                bufferedArray[j] = np.append(bufferedArray[j], 0)
        raw[i] = bufferedArray # Remove sync channel from raw data
        if i == len(trials['stimOn_times'])-1: break
    sr.close()

    meta = {
        'fs': sr.fs,
        'trialCount': len(stimOn_times),
        'timeWin': timeWin,
        'acronym': acronym,
        'regionId': brainLocationId_ccf_2017[0],
        'eid': eid,
        'pids': pids,
        'channels': channels,
        'invalidTrials': invalidTrials
    }

    dataCompressed = np.empty((meta['trialCount'], round(meta['fs'] )* (timeWin + 1)))
    for i, e in enumerate(raw):
        dataCompressed[i] = np.mean(e, axis=0)

    if saveResult:
        Path(f'{basePath}{acronym.replace("/", "_")}/ps').mkdir(parents=True, exist_ok=True)
        np.save(dataFile, dataCompressed)
        with open(metaFile, 'wb') as f:
            pickle.dump(meta, f)

    gc.collect()

    if removeInvalidTrials:
        dataCompressed, times, contrast, feedback_type = __removeInvalidTrials(dataCompressed, times, feedback_type, contrast, meta)

    # data shape: (no. of trials, no. of datapoints)
    # times: (time of stimOn in s, time of stimOff in s, time of feedback in s, first movement time in s (reaction time), feedback time in s)
        # time of stimOn, stimOff & feedback = time since beginning of the recording; first movement & feedback time = time since stimOn for this specific trial
    # feedback_type: 1 if the animal turned the wheel to the correct side, -1 if the animal turned the wheel to the incorrect side
    # contrast: normalized contrast values, negative numbers indicate left side, positive numbers indicate right side, -2 or 2 indicate 'catch' trial
    # meta: info about recording
    return dataCompressed, times, feedback_type, contrast, meta

def getSpecificTrials(pid: str, acronym: str, invalidTrials: bool | str = 'both', stimSide: str = 'both', contrastValues: list = [0.0625, 0.125, 0.25, 1, 2], onlyLocalSessions: bool = False) -> tuple:
    """Returns trials of session which fit the given conditions

    Parameters
    ----------
    pid: str
        The probe ID of the probe
    acronym: str
        The acronmy of the brain region
    invalidTrials: bool | str, optional
        Should invalid trials be considered? True = yes, False = no, 'both' = valid & invalid trials will be considered (default is 'both')
    stimSide: str, optional
        Which side should the stimulus appear on, 'left', 'right', or 'both' (default is 'both')
    contrastValues: list, optional
        Which contrast values should be considered? Note: only positive contrast values will be considered (default is [0.0625, 0.125, 0.25, 1, 2])
    onlyLocalSessions: bool, optional
        If the search should only consider locally saved sessions (default is False)

    Returns
    -------
    data: list
        See function 'getDataFromPid(...)'
    times: tuple
    feedback: list
    contrast: list
    meta: dict

    Raises
    ------
    Invalid value for parameter
       Parameters invalidTrials or stimSide received invalid values
    Invalid contrastValues
        Given contrastValues not in [0.0625, 0.125, 0.25, 1, 2]
    """
    
    data, times, feedback, contrast, meta = getDataFromPid(pid=pid, acronym=acronym, removeInvalidTrials=False, onlyLocal=onlyLocalSessions)

    if onlyLocalSessions and len(data) == 0:
        return [[] for i in range(5)]

    if invalidTrials == True:
        data, times, contrast, feedback = __getInvalidTrials(data, times, feedback, contrast, meta)
    elif invalidTrials == False:
        data, times, contrast, feedback = __removeInvalidTrials(data, times, feedback, contrast, meta)
    elif invalidTrials != 'both':
        raise Exception("Invalid value for parameter 'invalidTrials'")

    if stimSide == 'left':
        indices = np.where(contrast < 0)[0]
        data = data[indices]
        times = __getTimesForIndices(times, indices)
        contrast = contrast[indices]
        feedback = feedback[indices]
    elif stimSide == 'right':
        indices = np.where(contrast > 0)[0]
        data = data[indices]
        times = __getTimesForIndices(times, indices)
        contrast = contrast[indices]
        feedback = feedback[indices]
    elif stimSide != 'both':
        raise Exception("Invalid value for parameter 'stimSide'")

    if len(contrastValues) == 0:
        raise Exception("Invalid contrastValues")

    allIndices = []
    for x in contrastValues:
        if abs(x) not in [0.0625, 0.125, 0.25, 1, 2]:
            raise Exception("Invalid contrastValues")
        allIndices = allIndices + list(np.where(abs(contrast) == abs(x))[0])

    data = data[allIndices]
    times = __getTimesForIndices(times, allIndices)
    contrast = contrast[allIndices]
    feedback = feedback[allIndices]

    return data, times, feedback, contrast, meta

def __getTimesForIndices(times: tuple, indices: list) -> tuple:
    return (times[0][indices], times[1][indices], times[2][indices], times[3][indices], times[4][indices])

def __removeInvalidTrials(data: list, times: tuple, feedbackType: list, contrast: list, meta: dict) -> tuple:
    dataInvalidTrialsRemoved = np.array([x for i, x in enumerate(data) if i not in meta['invalidTrials']])
    contrastInvalidTrialsRemoved = np.array([x for i, x in enumerate(contrast) if i not in meta['invalidTrials']])

    indices = [i for i, x in enumerate(feedbackType) if x > 0 and i < len(dataInvalidTrialsRemoved)]
    dataWithoutFails = dataInvalidTrialsRemoved[indices]
    contrastWithoutFails = contrastInvalidTrialsRemoved[indices]
    feedbackWithoutFails = feedbackType[indices]

    t = (times[0][indices], times[1][indices], times[2][indices], times[3][indices], times[4][indices])

    return dataWithoutFails, t, contrastWithoutFails, feedbackWithoutFails

def __getInvalidTrials(data: list, times: tuple, feedbackType: list, contrast: list, meta: dict) -> tuple:
    dataInvalidTrialsRemoved = np.array([x for i, x in enumerate(data) if i not in meta['invalidTrials']])
    contrastInvalidTrialsRemoved = np.array([x for i, x in enumerate(contrast) if i not in meta['invalidTrials']])

    indices = [i for i, x in enumerate(feedbackType) if x < 0 and i < len(dataInvalidTrialsRemoved)]
    dataWithFails = dataInvalidTrialsRemoved[indices]
    contrastWithFails = contrastInvalidTrialsRemoved[indices]
    feedbackWithFails = feedbackType[indices]
    t = (times[0][indices], times[1][indices], times[2][indices], times[3][indices], times[4][indices])

    return dataWithFails, t, contrastWithFails, feedbackWithFails

def __getFourierCoefficient(data, hz, sampleRate) -> complex:
    n = len(data)
    fftFreqs = np.fft.fftfreq(n, 1 / float(sampleRate))
    freqIdx = np.argmin(np.abs(fftFreqs - hz))
    fft = np.fft.fft(data)
    return fft[freqIdx]

def getLaggedCoherence(data: list, hz: float, cycles: int, sampleRate = 2500, shift: bool = False) -> list:
    """Returns the lagged coherence for the given number of cycles and frequency

    Parameters
    ----------
    data: list
        The input data. Shape: (# of trials, # of datapoints per trial)
    hz: float
        The frequency of interest
    cycles: int
        The number of cycles
    sampleRate: int, optional
        The sample rate of the data (default is 2500)
    shift: bool, optional
        If the method should return the shifted lagged coherence
        in addition to the non-shifted result. Shifting factors are [2/3, 1/3, 1/4, 0 (non-shifted)] in this order (default is False)
    
    Returns
    -------
    shiftedResult: float
        If shift == False: Returns list with a single lagged coherence value for given parameters
        If shift == True: Returns four lagged coherence values, three shifted values & one non-shifted value in the following order: [2/3, 1/3, 1/4, 0 (non-shifted)]
    """
    windowLen = (1/hz) * cycles
    nSamples = int(sampleRate * windowLen)

    if shift:
        shifts = [2/3, 1/3, 1/4, 0]
    else:
        shifts = [0]

    chunks = [[] for i in shifts]
    for ind, shift in enumerate(shifts):
        shiftLen = int((1/hz) * cycles * sampleRate * shift)
        for i in range(0 + shiftLen, len(data), nSamples):
            chunks[ind].append(data[i:i + nSamples])

    shiftedResult = [[] for i in shifts]

    for index, shift in enumerate(shifts):
        nChunks = len(chunks[index])

        fftCoefs = np.zeros(nChunks, dtype=complex)
        for ind, chunk in enumerate(chunks[index]):
            hannWindow = np.hanning(len(chunk))
            fftCoefs[ind] = __getFourierCoefficient(chunk * hannWindow, hz, sampleRate)

        numerator = np.sum(fftCoefs[:-1] * np.conj(fftCoefs[1:]))
        denominator = np.sqrt(np.sum(np.abs(fftCoefs[:-1])**2) * np.sum(np.abs(fftCoefs[1:])**2))

        shiftedResult[index] = np.abs(numerator / denominator)

    return shiftedResult

def getContrastThetaPowerCorrelation(pids: list, regions: list, saveResult: bool = False) -> tuple:
    """Returns the correlation between the mean theta power and the trial contrast

    Parameters
    ----------
    pids: list
        List of pids from which the mean theta should be computed
    regions: list
        List of regions. Note: if len(regions) < len(pids), the first region of the list will be assumed for every pid
    saveResult: bool, optional
        If the computed mean theta power should be saved (default is False)

    Returns
    -------
    t: list
        Time, ranging from -0.5s to +2.5s
    r: list
        Pearson correlation coefficients for every tiem point
    p-values: list
        p-values for every time point computed using the binomial test on the pearson p-values
    """

    if len(regions) < len(pids):
        regions = np.full(len(pids), regions[0])

    windowLength = 2500
    hanning = np.hanning(windowLength)
    hanningOverl = len(hanning) - 1

    correlationCoefficient = np.zeros((len(pids), 7501))
    pValues = np.zeros((len(pids), 7501))

    for index, pid in enumerate(pids):
        try:
            data, times, feedbackType, contrast, meta = getDataFromPid(pid, regions[index], removeInvalidTrials=True)
        except:
            continue

        specFile = f'{basePath}{regions[index].replace("/", "_")}/spec/{pid}-{regions[index].replace("/", "_")}-spec.npy'
        if os.path.isfile(specFile):
            with open(specFile, 'rb') as f:
                meanThetaPerTrial = np.load(f)
                for i, trial in enumerate(data):
                    if np.abs(contrast[i]) == 2:
                        meanThetaPerTrial[i] = 0
        else:
            a = np.empty((len(data), 5, int(round(meta['fs']) * meta['timeWin']) + 1), dtype=np.single)

            for i, trial in enumerate(tqdm(data)):
                _, _, Sxx = signal.spectrogram(trial, meta['fs'], hanning, len(hanning), hanningOverl)
                a[i] = Sxx[6:11]
            meanThetaPerTrial = np.mean(a, axis=1)

            for i, trial in enumerate(data):
                if np.abs(contrast[i]) == 2:
                    meanThetaPerTrial[i] = 0

            if saveResult:
                Path(f'{basePath}{regions[index].replace("/", "_")}/spec').mkdir(parents=True, exist_ok=True)
                np.save(specFile, meanThetaPerTrial)

        contrastValues = sorted(set(contrast))[1:-1]
        thetaForContrast = [[] for i in range(len(contrastValues))]

        for i, trial in enumerate(meanThetaPerTrial):
            contrastIdx = np.where(contrastValues == contrast[i])[0]
            if len(contrastIdx) == 0:
                continue
            contrastIdx = contrastIdx[0]
            thetaForContrast[contrastIdx].append(trial)

        meanThetaForContrast = [[] for i in range(len(contrastValues))]
        for i, contrastTheta in enumerate(thetaForContrast):
            meanThetaForContrast[i] = np.mean(contrastTheta, axis=0)

        for i in range(len(meanThetaForContrast[0])):
            coeff, pValue = pearsonr(contrastValues, np.array(meanThetaForContrast)[:,i])
            pValues[index][i] = pValue
            correlationCoefficient[index][i] = coeff

        gc.collect()

    return np.arange(-0.5, 2.5, 1/2500), np.mean(correlationCoefficient, axis=0)[:-1], __binomTest(pValues)[:-1]


def getReactionTimeThetaPowerCorrelation(pids: list, regions: list, saveResult: bool = False) -> tuple:
    """Returns the correlation between mean theta power and reaction time

    Parameters
    ----------
    pids: list
        List of pids from which the mean theta should be computed
    regions: list
        List of regions. Note: if len(regions) < len(pids), the first region of the list will be assumed for every pid
    saveResult: bool, optional
        If the computed mean theta power should be saved (default is False)

    Returns
    -------
    t: list
        Time, ranging from -0.5s to +2.5s
    r: list
        Pearson correlation coefficients for every tiem point
    p-values: list
        p-values for every time point computed using the binomial test on the pearson p-values
    """
    if len(regions) < len(pids):
        regions = np.full(len(pids), regions[0])

    windowLength = 2500
    hanning = np.hanning(windowLength)
    hanningOverl = len(hanning) - 1

    correlationCoefficient = np.zeros((len(pids), 7501))
    pValues = np.zeros((len(pids), 7501))

    for index, pid in enumerate(pids):
        try:
            data, times, feedbackType, contrast, meta = getDataFromPid(pid, regions[index], removeInvalidTrials=True)
        except:
            continue
        
        reactionTimes = times[3]

        specFile = f'{basePath}{regions[index].replace("/", "_")}/spec/{pid}-{regions[index].replace("/", "_")}-spec.npy'
        if os.path.isfile(specFile):
            with open(specFile, 'rb') as f:
                meanThetaPerTrial = np.load(f)
        else:
            a = np.empty((len(data), 5, int(round(meta['fs']) * meta['timeWin']) + 1), dtype=np.single)
            
            for i, trial in enumerate(tqdm(data)):
                _, _, Sxx = signal.spectrogram(trial, meta['fs'], hanning, len(hanning), hanningOverl)
                a[i] = Sxx[6:11]
            meanThetaPerTrial = np.mean(a, axis=1)

            if saveResult:
                Path(f'{basePath}{regions[index].replace("/", "_")}/spec').mkdir(parents=True, exist_ok=True)
                np.save(specFile, meanThetaPerTrial)
        
        nanIndex = np.argwhere(np.isnan(reactionTimes)).flatten()
        reactionTimes = np.delete(reactionTimes, nanIndex)

        for i in range(len(meanThetaPerTrial[0])):
            coeff, pValue = pearsonr(reactionTimes, np.delete(np.array(meanThetaPerTrial)[:,i], nanIndex))
            pValues[index][i] = pValue
            correlationCoefficient[index][i] = coeff

        gc.collect()

    return np.arange(-0.5, 2.5, 1/2500), np.mean(correlationCoefficient, axis=0)[:-1], __binomTest(pValues)[:-1]

def __binomTest(pValues: list) -> list:
    result = np.zeros(len(pValues[0]))
    for i in range(len(pValues[0])):
        col = pValues[:,i]
        n = len(col)
        k = len(np.where(col < 0.05)[0])
        res = binomtest(k, n, 0.05)
        result[i] = getattr(res, 'pvalue')
    return result

def getLRThetaPowerKStestResult(pids: list, regions: list, windowLen: int = 0, saveResult: bool = False) -> tuple:
    """Return the KS-test of left and right mean theta power

    Parameters
    ----------
    pids: list
        List of pids from which the mean theta should be computed
    regions: list
        List of regions. Note: if len(regions) < len(pids), the first region of the list will be assumed for every pid
    windowLen: int, optional
        Length of the window for the window averaging function. If windowLen = 0, no averaging is applied (default is 0)
    saveResult: bool, optional
        If the computed mean theta power should be saved (default is False)

    Returns
    -------
    t: list
        Time, ranging from -0.5s to +2.5s
    percentage: list
        Percent of KS-test results below the threshold of 0.05
        KS-test is performed on the left mean theta vs. right mean theta of all sessions provided
    """
    if len(regions) < len(pids):
        regions = np.full(len(pids), regions[0])

    windowLength = 2500
    hanning = np.hanning(windowLength)
    hanningOverl = len(hanning) - 1

    allTrialsKSTestResults = np.zeros((len(pids), 7501))

    for index, pid in enumerate(pids):
        try:
            data, times, feedbackType, contrast, meta = getDataFromPid(pid, regions[index], removeInvalidTrials=True)
        except:
            continue

        leftInd = np.where(np.logical_and(contrast < 0, contrast > -2))
        rightInd = np.where(np.logical_and(contrast > 0, contrast < 2))

        leftMeanTheta = np.empty(len(leftInd))
        rightMeanTheta = np.empty(len(rightInd))

        specFile = f'{basePath}{regions[index].replace("/", "_")}/spec/{pid}-{regions[index].replace("/", "_")}-spec.npy'
        if os.path.isfile(specFile):
            with open(specFile, 'rb') as f:
                meanThetaPerTrial = np.load(f)
        else:
            a = np.empty((len(data), 5, int(round(meta['fs']) * meta['timeWin']) + 1), dtype=np.single)

            for i, trial in enumerate(tqdm(data)):
                _, _, Sxx = signal.spectrogram(trial, meta['fs'], hanning, len(hanning), hanningOverl)
                a[i] = Sxx[6:11]
            meanThetaPerTrial = np.mean(a, axis=1)

            if saveResult:
                Path(f'{basePath}{regions[index].replace("/", "_")}/spec').mkdir(parents=True, exist_ok=True)
                np.save(specFile, meanThetaPerTrial)

        leftMeanTheta = meanThetaPerTrial[leftInd]
        rightMeanTheta = meanThetaPerTrial[rightInd]

        trialKSTestResult = np.zeros(len(meanThetaPerTrial[0]))

        for i in range(len(meanThetaPerTrial[0])):
            left = leftMeanTheta[:,i]
            right = rightMeanTheta[:,i]
            leftHist = np.histogram(left, bins=100)[0]
            rightHist = np.histogram(right, bins=100)[0]

            res = kstest(leftHist, rightHist)

            trialKSTestResult[i] = res[1]
            
        allTrialsKSTestResults[index] = trialKSTestResult

    percentage = np.zeros(7501)
    for i in range(len(allTrialsKSTestResults[0])):
        percentage[i] = len(allTrialsKSTestResults[np.where(allTrialsKSTestResults[:,i] < 0.05)]) / len(allTrialsKSTestResults)

    percentage = percentage[:-1]

    if windowLen > 0:
        result = []
        for i in range(0, len(percentage), windowLen):
            result.append(np.mean(percentage[i:i+windowLen]))
        return np.arange(-0.5, 2.5, 1/2500*windowLen), result
    else:
        return np.arange(-0.5, 2.5, 1/2500), percentage

def generateSpectrogramForData(data: list, meta: dict, plot: bool = False, title: str = "") -> tuple:
    """Returns a spectrogram for the data

    Paramters
    ---------
    data: list
        The input data. Shape: (# of trials, # of datapoints per trial)
    meta: dict
        The metadata for the given data. Used for sample rate & time window of data
    plot: bool, optional
        If the result should be plotted (default is False)
    title: str, optional
        The title of the plot, only relevant if plot = True (default is "")

    Returns
    -------
    mainSegmentTimes: list
        Times of the windows
    mainF: list
        List with all frequencies (0 to 100)
    Sxx: list
        The spectrogram itself
    """
    windowLength = 2500
    highestFrequency = 100

    a = np.empty((len(data), highestFrequency + 1, int(round(meta['fs']) * (meta['timeWin']) + 1)), dtype=np.single)

    hanning = np.hanning(windowLength)
    hanningOverlap = len(hanning) - 1
    mainF = 0
    mainSegmentTimes = 0
    for i in tqdm(range(len(data))):
        mainF, mainSegmentTimes, Sxx = signal.spectrogram(data[i], meta['fs'], hanning, len(hanning), hanningOverlap)
        a[i] = Sxx[0:highestFrequency + 1]

    mainSegmentTimes -= 1
    mainF = mainF[0:highestFrequency+1]
    Sxx = np.log10(np.mean(a, axis=0))

    if plot:
        plt.figure(figsize=(5, 10))
        plt.pcolormesh(mainSegmentTimes, mainF, Sxx, cmap='viridis', vmin=np.min(Sxx), vmax=np.max(Sxx))
        plt.axvline(0, color='k', linewidth=2, zorder=2)
        plt.yticks(np.arange(0, 100, 5))
        plt.ylim((0, 100))
        cbar = plt.colorbar()
        cbar.set_label('Logarithmic power')
        if len(title) > 0: plt.title(title)
        plt.xlabel('Time in seconds since go cue')
        plt.ylabel('Frequency in hertz')
        plt.show()

    return mainSegmentTimes, mainF, Sxx

def generatePowerSpectrumForData(data: list, meta: dict, plot: bool = False, title: str = "") -> tuple:
    """Returns a power spectrum for the data

    Parameters
    ----------
    data: list
        The input data. Shape: (# of trials, # of datapoints per trial)
    meta: dict
        The metadata for the given data. Used for sample rate & time window of data
    plot: bool, optional
        If the result should be plotted (default is False)
    title: str, optional
        The title of the plot, only relevant if plot = True (default is "")

    Returns
    -------
    ps: list
        The power spectrum itself
    freqs: list
        The list of frequencies corresponding to each entry in the ps list
    """
    ps = np.empty((len(data), len(data[0])))

    for i in range(len(data)):
        ps[i] = np.abs(np.fft.fft(data[i]))**2
    timeStep = 1 / meta['fs']
    freqs = np.fft.fftfreq(data[0].size, timeStep)
    idx = np.argsort(freqs)

    ps = np.mean(np.log10(ps), axis=0)[idx]

    if plot:
        plt.figure()
        plt.plot(freqs[idx], ps)
        plt.xticks(np.arange(0, 100, 5))
        plt.grid()
        plt.ylabel("Logarithmic power")
        plt.xlabel("Frequency in Hz")
        if len(title) > 0: plt.title(title)
        plt.xlim(0, 100)
        plt.show()

    return ps, freqs[idx]

def __fitAperiodic(freqs, powerSpectrum, noLog: bool = False) -> tuple:
    def expo_func(xs, offset, exp):
        return offset - np.log10(xs**exp)
    
    frequencies = np.arange(1, 100.25, 0.25)

    reducedSpectrum = np.empty(len(frequencies))
    for i, e in enumerate(frequencies):
        index = np.where(freqs == e)
        if noLog:
            reducedSpectrum[i] = powerSpectrum[index]
        else:
            reducedSpectrum[i] = np.log10(powerSpectrum[index])
    
    popt, pcov = curve_fit(expo_func, frequencies, reducedSpectrum)
    initialFit = expo_func(frequencies, popt[0], popt[1])
    
    flatspec = reducedSpectrum - initialFit
    flatspec[flatspec < 0] = 0
    perc_thresh = np.percentile(flatspec, 0.025)
    perc_mask = flatspec <= perc_thresh
    freqs_ignore = frequencies[perc_mask]
    spectrum_ignore = reducedSpectrum[perc_mask]
    
    finalPopt, finalPcov = curve_fit(expo_func, freqs_ignore, spectrum_ignore, p0=popt)
    finalFit = expo_func(frequencies, finalPopt[0], finalPopt[1])

    return frequencies, reducedSpectrum, finalFit

def parameterizePowerSpectra(ps: list, containsLog: bool = False, reFit: bool = False) -> tuple:
    """Returns the modeled power spectra

    Parameters
    ----------
    ps: list
        List of the original power spectra
    reFit: bool, optional
        If the aperiodic component should be re-fitted or not (default is False)
        [We discovered that the model is more accurate without the re-fitting step]

    Returns
    -------
    frequencies: list
        List of frequencies corresponding to a single power spectrum
    allPs: list
        List of all modeled power spectra (allPs[i] == model of ps[i])
    reducedPs: list
        List of the input list ps, but with a reduced number of data points
    """
    fm = FOOOF()

    idx = [i for i, x in enumerate(ps) if np.mean(x) == 0]
    ps = np.delete(ps, idx, axis=0)

    freqs = np.arange(-1250, 1250, 0.25)

    def gaussian(x, center, height, width):
        return height * np.exp(-(x - center)**2 / (2 * width**2))

    allPs = []
    reducedPs = []

    for el in ps:
        if containsLog:
            el = np.power(10, el)
        fm.fit(freqs, el, [1, 100])
        allGaussian = fm.gaussian_params_
        frequencies, reducedSpectrum, firstFit = __fitAperiodic(freqs, el)

        mainGauss = np.zeros(len(frequencies))
        for i in allGaussian:
            mainGauss += gaussian(frequencies, i[0], i[1], i[2])
        
        if reFit:
            _, _, finalFit = __fitAperiodic(frequencies, reducedSpectrum - mainGauss, noLog=True)
            finalSpectrum = mainGauss + finalFit
        else:
            finalSpectrum = mainGauss + firstFit

        allPs.append(finalSpectrum)
        reducedPs.append(reducedSpectrum)

    return frequencies, allPs, reducedPs