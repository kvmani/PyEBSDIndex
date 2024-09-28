# Radon indexing of a demo dataset

# if installed from conda or pip, this is likely not necessary, but if installed from source, or using a developer branch, this can be quite useful.
# import sys
# sys.path.insert(0, "/Path/to/PyEBSDIndex")

import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
from pyebsdindex import tripletvote, ebsd_pattern, ebsd_index, ebsdfile, pcopt
from pyebsdindex.EBSDImage import IPFcolor

# An example of indexing a file of patterns.
file = '/Path/to/example.up1'  # or ebsp, or h5oina or Bruker h5
PC = np.array([0.46, 0.70, 0.64])  # this is pulled from the .ang/ctf/h5 file, but only is a rough guess. We will refine it later.
cam_elev = 5.3  # The tilt of the camera from horizontal -- positive angles are tilted below the horizontal.
sampleTilt = 70.0  # sample tilt
vendor = 'EDAX'  # notes the conventions for pattern center and orientations.

# Set up some phases. There are shortcuts for common phases (FCC, BCC, HCP).
# For the first phase, we will use the shortcut method for FCC.
fcc = tripletvote.addphase(libtype='FCC')

# It is possible to override the defaults for any of the parameters and to set a phase name.
austenite = tripletvote.addphase(libtype='FCC', phasename='Austenite', latticeparameter=[0.355, 0.355, 0.355, 90, 90, 90])

# For a non-shortcut phase like BCC:
ferrite = tripletvote.addphase(
    phasename='Ferrite',
    spacegroup=229,
    latticeparameter=[0.286, 0.286, 0.286, 90, 90, 90],
    polefamilies=[[0, 1, 1], [0, 0, 2], [1, 1, 2], [0, 1, 3]]
)

# Put the phases into a list.
phaselist = [austenite, ferrite]

# Alternatively, you can define the phases lazily:
phaselistlazy = [austenite, 'BCC', 'HCP']

# Define the radon and indexing parameters.
nT = 180  # degree resolution
nR = 90  # number of bins in rho space
tSig = 2.0  # gaussian kernel size in theta
rSig = 2.0  # gaussian 2nd derivative in rho
rhomask = 0.1  # fraction of radius to not analyze
backgroundsub = False  # background correction
nbands = 8

# # Initialize the indexer object.
# dat1, bnd1, indxer = ebsd_index.index_pats(
#     filename=file,
#     patstart=0, npats=1000,
#     return_indexer_obj=True,
#     backgroundSub=backgroundsub,
#     nTheta=nT, nRho=nR,
#     tSigma=tSig, rSigma=rSig,
#     rhoMaskFrac=rhomask, nBands=nbands,
#     phaselist=phaselist,
#     PC=PC, camElev=cam_elev,
#     sampleTilt=sampleTilt,
#     vendor=vendor,
#     verbose=2
# )
# imshape = (indxer.fID.nRows, indxer.fID.nCols)
#
# # Refine the PC guess
# startcolrow = [int(imshape[1] // 2) - 2, int(imshape[0] // 2) - 2]
# fID = ebsd_pattern.get_pattern_file_obj(file)
# pats, xyloc = fID.read_data(returnArrayOnly=True, convertToFloat=True, patStartCount=[startcolrow, [5, 5]])
# newPC = pcopt.optimize(pats, indxer, PC0=PC)
# indxer.PC = newPC
# print(newPC)
#
# # Index the entire file using multiple CPUs and GPUs.
# data, bnddata = ebsd_index.index_pats_distributed(
#     filename=file, patstart=0, npats=-1,
#     ebsd_indexer_obj=indxer, ncpu=18, verbose=2
# )
#
# # Display the results as an IPF map.
# ipfim = IPFcolor.makeipf(data, indxer)
# plt.imshow(ipfim)
#
# # Display fit and pattern quality
# fit = (data[-1]['fit']).reshape(imshape[0], imshape[1])
# plt.imshow(fit.clip(0, 2.0))
#
# pq = (data[-1]['pq']).reshape(imshape[0], imshape[1])
# plt.imshow(pq)
#
# # Writing data out
# ebsdfile.writeoh5(filename='MyScanData.oh5', indexer=indxer, data=data)
# ebsdfile.writeang(filename='MyScanData.ang', indexer=indxer, data=data)
#
# # Example of indexing an array of patterns.
# startcolrow = [10, 5]
# ncol = 200
# nrow = 300
# f = ebsd_pattern.get_pattern_file_obj(file)
# pats, xyloc = f.read_data(returnArrayOnly=True, convertToFloat=True, patStartCount=[startcolrow, [ncol, nrow]])
# print(pats.shape)
# print(pats.dtype)
# plt.imshow(pats[0, :, :], cmap='gray')
#
# # Indexing small arrays
# datasm, bnddatsm = ebsd_index.index_pats(patsin=pats, ebsd_indexer_obj=indxer, verbose=2)
#
# # Distributed indexing of large arrays
# datasm, bnddatsm = ebsd_index.index_pats_distributed(patsin=pats, ebsd_indexer_obj=indxer, ncpu=12)
#
# # Display IPF map for indexed array
# ipfim = IPFcolor.makeipf(datasm, indxer, xsize=200)
# plt.imshow(ipfim)
#
# # Indexing a single pattern
# pat1 = pats[0, :, :]
# print(pat1.shape)
# plt.imshow(pat1, cmap='gray')
#
# dat1, bnddat1 = ebsd_index.index_pats(patsin=pat1, ebsd_indexer_obj=indxer, verbose=2)
# dat1 = dat1[-1]
# print(dat1.dtype.names)
# print(dat1)
#
# # Indexing a single pattern on the CPU
# indxerCPU = copy.deepcopy(indxer)
# indxerCPU.bandDetectPlan.useCPU = False
# dat1, bnddat1 = ebsd_index.index_pats(patsin=pat1, ebsd_indexer_obj=indxerCPU, verbose=2)
# dat1 = dat1[-1]

# Indexing patterns from an HDF5 file
#h5file = r'C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\test_8x8.oh5'
h5file = r'C:\Users\kvman\Documents\ml_data\kikuchi_super_resolution\Med_Mn_10k_8x8.oh5'
f = h5py.File(h5file, 'r')
#h5pats = f['/map20240608160135320/EBSD/Data/Pattern']
h5pats = f['/map20240608140723780/EBSD/Data/Pattern'] ###Med_Mn_10K_8X8.oh5
h5data, h5bnddata, indxer = ebsd_index.index_pats(
    patsin=h5pats[0:1000, :, :],
    patstart=0, npats=1000,
    return_indexer_obj=True,
    backgroundSub=backgroundsub,
    nTheta=nT, nRho=nR,
    tSigma=tSig, rSigma=rSig,
    rhoMaskFrac=rhomask, nBands=nbands,
    phaselist=phaselist,
    PC=PC, camElev=cam_elev,
    sampleTilt=sampleTilt,
    vendor=vendor,
    verbose=2
)

# Index all patterns in the HDF5 file
#h5data, h5banddata = ebsd_index.index_pats_distributed(patsin=h5pats, ebsd_indexer_obj=indxer, ncpu=28)
#h5data, h5banddata = ebsd_index.index_pats(patsin=h5pats, ebsd_indexer_obj=indxer, ncpu=28)
h5data, h5banddata = ebsd_index.index_pats(patsin=h5pats, ebsd_indexer_obj=indxer, verbose=2)
ebsdfile.writeoh5(filename='MyScanData.oh5', indexer=indxer, data=h5data)

# # Display IPF map for indexed array
ipfim = IPFcolor.makeipf(h5data, indxer, xsize=200)
plt.imshow(ipfim)
plt.show()