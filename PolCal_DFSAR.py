#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:59:03 2026

@author: hephzy
"""

from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import rasterio as rio
import xml.etree.ElementTree as ET
import csv

# computes channel imbalances (f_r, f_t) and cross talk terms (𝛿1^r, 𝛿1^t, 𝛿2^r, 𝛿2^t) for each cell in the DFSAR image -- NEEDS WORK
# follows Sun et al. 2018 procedure
# applies f_r, f_t, 𝛿1^r, 𝛿1^t, 𝛿2^r, 𝛿2^t to polarimetircally calibrate given image
# output: png of histograms comparing uncalibrated and calibrated phase differences (similar to fig. 4 in Bhiravarasu et al. 2022)

home = '/projects/cos-lab-iganesh3/proj/lunar_dfsar/data/' # CEDAR on Bumi
folder = home + "ch2_sar_ncxl_20210824t163808566_d_fp_d18/data/calibrated/20210824/"
all_files = os.listdir(folder)

out_folder = folder+"derived/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# read only SLC files 'sli'
patterns = ["sli_xx_fp_vv", "sli_xx_fp_hh", "sli_xx_fp_hv", "sli_xx_fp_vh"]
#files = [file for file in files if file.endswith('.tif')] # keeping only tiff files
files = [file for file in all_files if file.endswith('.tif') and any(pattern in file for pattern in patterns)] # keeping only SLI files

# xml files
sli_xml = [file for file in all_files if file.endswith('.xml') and "sli_xx_fp_xx" in file][0] 
gri_xml = [file for file in all_files if file.endswith('.xml') and "gri_xx_fp_xx" in file][0] 

def readSLC(file):
    print("Reading " + file+"----------")
    with rio.open(folder+file) as src:
        meta = src.meta # Get the metadata of the input file to preserve it for the output
        
        # Read I and Q images
        I = src.read(1).astype(np.float32)  # Real (I) band
        Q = src.read(2).astype(np.float32) # imaginary (Q) band
        complex_DN = I + 1j * Q # complex pixel values: DN_complex = I + Qi
        DN = np.abs(I + 1j * Q) # Magnitude of DN
        phase = np.angle(complex_DN) # [rad] ; -180 to 180
        
        print(f"DN: {np.nanmin(complex_DN):.3f} to {np.nanmax(complex_DN):.3f}")
        print(f"Phase: {np.nanmin(phase):.3f} to {np.nanmax(phase):.3f} rad")
        
        return complex_DN
    
def print_meta(file):
    with rio.open(folder+file) as src:
        print(f"{file}: Shape = {src.height} x {src.width} ; bands = {src.count}")
        
    
def get_meta(file):
    with rio.open(folder+file) as src:
        meta = src.meta
        return meta

# Radiometrically calibrate SLC images using XML constants ---------------------
def RadCal_SLC(file, xml_file):
    # parse calibration constant from XML
    tree = ET.parse(folder + xml_file) # Parse sli XML file
    root = tree.getroot()
    K_dB = np.float32(root.find('.//{*}calibration_constant').text) # Extract the calibration constant value
    NES0 = -27.9 # Postlaunch NESZ Value for signal power normalized to 100 km altitude and at 30° angle of incidence (B2021)
    #print(f"Calibration Constant: {K_dB}")
    
    complex_DN = readSLC(file) # complex DN from SLC image
    
    # Radiometric calibration
    sigma0 = complex_DN / (10 ** (K_dB / 10.0)) # linear sigma0 
    sigma0_dB = 10 * np.log10(sigma0) # [dB]
    print(f"{file}: sigma0 [dB]: {np.nanmin(sigma0_dB):.3f} to {np.nanmax(sigma0_dB):.3f}")

    """# Save as GeoTIFF
    meta.update(dtype=rio.float32, nodata=np.nan) # Update metadata for the output
    with rio.open(folder+"derived/"+file[:-4]+"_sigma0.tif", 'w', **meta) as dst: 
        dst.write(sigma0.astype(rio.float32), 1)"""
        
    return sigma0

def getLookFactor(xml_file):
    # parse look factor from GRI XML
    tree = ET.parse(folder + xml_file) # Parse the XML file
    root = tree.getroot()
    N_rg = np.float32(root.find('.//{*}range_looks').text) # No. of looks along range
    N_az = np.float32(root.find('.//{*}azimuth_looks').text) # No. of looks along azimuth
    
    print(f"Azimuth looks: {N_az}, range looks: {N_rg}")
    
    return N_rg, N_az

def multilook(Mq, N_az, N_rg):
    # N_az, N_rg must be >= 1
    H, W = Mq.shape[2], Mq.shape[3]

    # trim to divisible size 
    H2 = (H // N_az) * N_az
    W2 = (W // N_rg) * N_rg
    Mq = Mq[:, :, :H2, :W2]

    Hm = H2 // N_az
    Wm = W2 // N_rg

    Ml = np.zeros((3, 3, Hm, Wm), dtype=Mq.dtype)

    for i in range(3):
        for j in range(3):
            Ml[i, j] = (Mq[i, j].reshape(Hm, N_az, Wm, N_rg).mean(axis=(1, 3)))

    return Ml

def plot_img(Mq, fname, clabel):
    
    Mq = 10 * np.log10(np.abs(Mq) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(Mq, cmap='viridis', alpha=1)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(clabel)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.savefig(out_folder+fname, dpi = 300, bbox_inches='tight')
    plt.close()
    
# Sun et al. 2018 procedure
def PolCal0(O_hh, O_hv, O_vh, O_vv, N_rg, N_az):
    O = np.array([O_hh, O_hv, O_vh, O_vv]) # observed scattering matrix
    rows, cols = O_hh.shape
    
    # step 0: compute covariance matrix of O
    #C = O @ O.conj().T # C = OO^H
    C = np.einsum('ikl,jkl->ijkl', O, O.conj())
    ## multiook covariance before proceeding ??
    C = multilook(C, N_az, N_rg) # keep N_ra = 1
    print(C.shape)
    
    # initiate cross talk terms m1, m2, m3, m4
    m0 = np.zeros((4,rows,cols), dtype=np.complex64) # all zeroes ; 4,rows,cols 
    
    return C, m0, rows, cols
    #W, m_new, diff_m = PolCal(C, m0, rows, cols)

# Cell wise polarimetic calibration coefficients computation -- needs to be debugged
def PolCal(C, m, rows, cols):
    # Unpack cov matrix
    C_hhhh = C[0,0] # diagonal elements of C 
    C_hvhv = C[1,1]
    C_vhvh = C[2,2]
    C_vvvv = C[3,3]
    
    C_hhhv = C[0,1]
    C_hhvh = C[0,2]
    C_hhvv = C[0,3]
    
    C_hvhh = C[1,0]
    C_hvvh = C[1,2]
    C_hvvv = C[1,3]
    
    C_vhhh = C[2,0]
    C_vhhv = C[2,1]
    C_vhvv = C[2,3]
    
    C_vvhh = C[3,0]
    C_vvhv = C[3,1]
    C_vvvh = C[3,2]
    
    # step 1: compute channel imbalances f_1, f_2 from C using eqn 14 
    f1_mod = (np.sqrt(C_vvvv/C_hhhh) / np.sqrt(C_hvhv/C_vhvh))**0.5
    f2_mod = (np.sqrt(C_vvvv/C_hhhh) * np.sqrt(C_hvhv/C_vhvh))**0.5
    
    phi_f1 = (np.angle(C_vvhh) +  np.angle(C_vhhv))/2
    phi_f2 = (np.angle(C_vvhh) -  np.angle(C_vhhv))/2
    
    #f1 = f1_mod * complex(np.cos(phi_f1), np.sin(phi_f1)) # channel imbalance of the receiver antenna ; f_r
    #f2 = f2_mod * complex(np.cos(phi_f2), np.sin(phi_f2)) # channel imbalance of the transmit antenna ; f_t
    f1 = f1_mod * np.exp(1j * phi_f1)
    f2 = f2_mod * np.exp(1j * phi_f2)
    
    # step 2: derive partial covariance matix Wp of the true scattering matrix S using eqn 12
    f1c = np.conj(f1) # conjugates
    f2c = np.conj(f2)
    
    abs_f1_sq = np.abs(f1)**2 # magnitudes squared
    abs_f2_sq = np.abs(f2)**2
    
    # partial covariance matrix Wp
    Wp = np.array([[C_hhhh, C_hhhv / f2c, C_hhvh / f1c, C_hhvv / (f1c * f2c)],
                   [C_hvhh / f2, C_hvhv / abs_f2_sq, C_hvvh / (f1c * f2), C_hvvv / (f1c * abs_f2_sq)],
                   [C_vhhh / f1, C_vhhv / (f1 * f2c), C_vhvh / abs_f1_sq, C_vhvv / (abs_f1_sq * f2c)],
                   [C_vvhh / (f1 * f2), C_vvhv / (f1 * abs_f2_sq), C_vvvh / (abs_f1_sq * f2), C_vvvv / (abs_f1_sq * abs_f2_sq)]
                   ])
    
    # step 3: Apply cross talk corrections and derive the complete form of W using eqns 15-18
    # eqn 17a
    X = np.zeros((4, rows, cols), dtype=np.complex64)
    P = np.zeros((4,4,rows,cols), dtype=np.complex64)
    Q = np.zeros((4,4,rows,cols), dtype=np.complex64)
    
    X[0] = f1c * f2c * Wp[1,0] * f1 * f2 - (Wp[1,0] - Wp[2,3])/2
    X[1] = f1c * f2c * Wp[2,3] * f1 * f2 + (Wp[1,0] - Wp[2,3])/2
    X[2] = f1c * f2c * Wp[2,0] * f1 * f2 - (Wp[2,0] - Wp[1,3])/2
    X[3] = f1c * f2c * Wp[1,3] * f1 * f2 + (Wp[2,0] - Wp[1,3])/2
    
    P[0,0] = f1c * f2c * Wp[3,0] * f1 * f2
    P[0,2] = f1c * f2c * Wp[0,0] * f1
    
    P[1,1] = f1c * f2c * Wp[0,3] * f2
    P[1,3] = f1c * f2c * Wp[3,3] * f1 * f2
    
    P[2,0] = f1c * f2c * Wp[0,0] * f2
    P[2,2] = f1c * f2c * Wp[3,0] * f1 * f2
    
    P[3,1] = f1c * f2c * Wp[3,3] * f1 * f2
    P[3,3] = f1c * f2c * Wp[0,3] * f1
    
    Q[0,0] = f1c * f2c * Wp[1,2] * f1 * f2
    Q[0,3] = f1c * f2c * Wp[1,1] * f1 * f2
    
    Q[1,1] = f2c * Wp[2,1] * f1 * f2
    Q[1,2] = f1c * Wp[2,2] * f1 * f2
    
    Q[2,0] = f1c * f2c * Wp[2,2] * f1 * f2
    Q[2,3] = f1c * f2c * Wp[2,1] * f1 * f2
    
    Q[3,1] = f2c * Wp[1,1] * f1 * f2
    Q[3,2] = f1c * Wp[1,2] * f1 * f2
    
    # eqn 17b
    A = np.zeros((8,8,rows,cols), dtype=np.float32)
    zX = np.zeros((8,rows,cols), dtype=np.float32) # RHS
    delta_ri = np.zeros((8,rows,cols), dtype=np.float32)

    A[0:4,0:4] = np.real(P + Q)
    A[0:4,4:8] = -np.imag(P - Q)
    A[4:8,0:4] = np.imag(P + Q)
    A[4:8,4:8] = np.real(P - Q)
    
    zX[0:4] = np.real(X)
    zX[4:8] = np.imag(X)

    for i in range(rows):
        for j in range(cols):
            Aij = A[:,:,i,j]
            zXij = zX[:,i,j]
            delta_ri[:,i,j] = np.linalg.solve(Aij, zXij)
            
    delta = (delta_ri[0:4] + 1j * delta_ri[4:8]) # 4,rows,cols 
    m = m + delta # 4,rows,cols ; incremented elements of cross talk matrix D
    
    # eqn 10 ; cross talk matrix
    D = np.ones((4,4,rows,cols), dtype=np.complex64)
    
    D[0,1] = m[3]
    D[0,2] = m[0]
    D[0,3] = m[0] * m[3]
    
    D[1,0] = m[2] / f2
    D[1,2] = m[0] * m[2] / f2
    D[1,3] = m[0]
    
    D[2,0] = m[1] / f1
    D[2,1] = m[1] * m[3] / f1
    D[2,3] = m[3]
    
    D[3,0] = m[1] * m[2] / (f1 * f2)
    D[3,1] = m[1] / f1
    D[3,2] = m[2] / f2
    
    # eqn 15
    Dinv = np.zeros_like(D)
    for i in range(rows):
        for j in range(cols):
            Dinv[:,:,i,j] = np.linalg.inv(D[:,:,i,j])
    Dinv_H = np.conj(Dinv).transpose(1,0,2,3)
    
    W = np.zeros_like(Wp) # Cov matrix with cross talk compensation
    for i in range(rows):
        for j in range(cols):
            W[:,:,i,j] = (Dinv[:,:,i,j] @ Wp[:,:,i,j] @ Dinv_H[:,:,i,j])
        
    # step 4: get updated f1 and f2 using terms in W (eqn 19)
    f1_w_mod = (np.sqrt(W[3,3]/W[0,0]) / np.sqrt(W[1,1]/W[2,2]))**0.5
    f2_w_mod = (np.sqrt(W[3,3]/W[0,0]) * np.sqrt(W[1,1]/W[2,2]))**0.5
    
    phi_w_f1 = (np.angle(W[3,0]) +  np.angle(W[2,1]))/2
    phi_w_f2 = (np.angle(W[3,0]) -  np.angle(W[2,1]))/2
    f1_w = f1_w_mod * np.exp(1j * phi_w_f1)
    f2_w = f2_w_mod * np.exp(1j * phi_w_f2)
        
    # step 5: update cross talk terms using eqn 22
    m1_new = m[0] * f1
    m2_new = m[1] / f1_w
    m3_new = m[2] / f2_w
    m4_new = m[3] * f2
    
    m_new = np.array([m1_new, m2_new, m3_new, m4_new]) # 4,rows,cols
    
    # step 6: check for stability
    diff_m = np.abs(m_new - m)
    
    return W, m_new, diff_m
    
def main():
    
    # SLC files
    hh_slc = [file for file in files if file.endswith('.tif') and 'hh' in file][0] # hh file
    hv_slc = [file for file in files if file.endswith('.tif') and 'hv' in file][0]
    vh_slc = [file for file in files if file.endswith('.tif') and 'vh' in file][0]
    vv_slc = [file for file in files if file.endswith('.tif') and 'vv' in file][0]
    
    # Radiometric calibration ; outputs observed scattering matrix O
    O_hh = RadCal_SLC(hh_slc, sli_xml) # linear sigma0 ; radimetrically calibrated complex DN from SLC image    
    O_hv = RadCal_SLC(hv_slc, sli_xml)
    O_vh = RadCal_SLC(vh_slc, sli_xml)
    O_vv = RadCal_SLC(vv_slc, sli_xml)
    
    # observed scattering matrix
    O = np.array([[O_hh, O_hv],
                  [O_vh, O_vv]]) 
    rows, cols = O_hh.shape
    
    """# polarimetric calibration
    N_rg, N_az = getLookFactor(gri_xml)
    N_rg, N_az = int(N_rg), int(N_az)
    C, m0, rows, cols = PolCal0(O_hh, O_hv, O_vh, O_vv, N_rg, N_az) # initiate with zero cross talk
    
    with open(out_folder+'polcal.csv','w') as csvfile:
        writer=csv.writer(csvfile, delimiter='\t',lineterminator='\n',)
        writer.writerow(['i', 'diff'])
        
        ni = 1000 # no of iterations
        si = np.zeros(ni)
        for i in range(ni):
            if i==0:
                W, m_new, diff_m = PolCal(C, m0, rows, cols)
                sum_diff_m = np.mean(diff_m)
                si[i] = sum_diff_m
                writer.writerow([str(i), str(sum_diff_m)])
            
            else:
                W, m_new, diff_m = PolCal(W, m_new, rows, cols)
                sum_diff_m = np.mean(diff_m)
                si[i] = sum_diff_m
                writer.writerow([str(i), str(sum_diff_m)])"""
    
    # Apply f_r, f_t, 𝛿1^r, 𝛿1^t, 𝛿2^r, 𝛿2^t to calibrate input images
    
    fr = 1.007299 - 0.003133j
    ft = 1.005099 - 0.002021j
    m1 = 0.005056 + 0.004458j
    m2 = 0.128799 - 0.026406j
    m3 = 0.104015 - 0.030022j
    m4 = 0.002068 + 0.002646j
    
    # receive path matrix
    R = np.array([[fr, m1],
                  [m2, 1]])
    
    # transmit path matrix
    T = np.array([[ft, m3],
                  [m4, 1]])
    
    # true scattering matrix
    K = 80 # from XML
    O = np.moveaxis(O, [0,1], [-2,-1])
    S = (1/K) * (np.linalg.inv(R) @ O @ np.linalg.inv(T))
    S = np.moveaxis(S, [-2,-1], [0,1])
    
    S_hh = S[0,0]
    S_hv = S[0,1]
    S_vh = S[1,0]
    S_vv = S[1,1]
    
    # phase 
    O_hh_ph = np.degrees(np.angle(O_hh))
    S_hh_ph = np.degrees(np.angle(S_hh))
    
    O_hv_ph = np.degrees(np.angle(O_hv))
    S_hv_ph = np.degrees(np.angle(S_hv))
    
    O_vh_ph = np.degrees(np.angle(O_vh))
    S_vh_ph = np.degrees(np.angle(S_vh))
    
    O_vv_ph = np.degrees(np.angle(O_vv))
    S_vv_ph = np.degrees(np.angle(S_vv))
    
    # observed inter-channel phases
    #phi_co_obs = np.degrees(np.angle(O_hh * np.conj(O_vv)))
    #phi_cx_obs  = np.degrees(np.angle(O_hv * np.conj(O_vh)))
    phi_co_obs = O_hh_ph - O_vv_ph
    phi_cx_obs = O_hv_ph - O_vh_ph
    
    # calibrated inter-channel phases
    #phi_co_cal = np.degrees(np.angle(S_hh * np.conj(S_vv)))
    #phi_cx_cal  = np.degrees(np.angle(S_hv * np.conj(S_vh)))
    phi_co_cal = S_hh_ph - S_vv_ph
    phi_cx_cal = S_hv_ph - S_vh_ph
    
    # histogram
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), dpi=300)
    bins = 200
    
    # HH
    axes[0].hist(phi_co_obs.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[0].hist(phi_co_cal.ravel(), bins=bins, alpha=0.5, label='Calibrated')
    axes[0].set_title('co-pol')
    axes[0].legend()
    
    # HV
    axes[1].hist(phi_cx_obs.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[1].hist(phi_cx_cal.ravel(), bins=bins, alpha=0.5, label='Calibrated')
    axes[1].set_title('cross-pol')
    axes[1].legend()
    
    """# VH
    axes[1,0].hist(O_vh_ph.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[1,0].hist(S_vh_ph.ravel(), bins=bins, alpha=0.5, label='Calibrated')
    axes[1,0].set_title('VH')
    axes[1,0].legend()
    
    # VV
    axes[1,1].hist(O_vv_ph.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[1,1].hist(S_vv_ph.ravel(), bins=bins, alpha=0.5, label='Calibrated')
    axes[1,1].set_title('VV')
    axes[1,1].legend()"""
    
    for ax in axes.ravel():
        ax.set_xlabel('Phase (deg)')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(out_folder+'ch2_sar_ncxl_20210824t163808566_d_fp_d18_phase_diff_polcal.png', dpi=300, bbox_inches='tight')
    plt.close()

main()