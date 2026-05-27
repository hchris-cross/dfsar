#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:19:23 2026

@author: hephzy
"""

# polarimetric calibration and orthorectification not included

from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import rasterio as rio
import xml.etree.ElementTree as ET

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
    print(f"Calibration Constant: {K_dB}")
    
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
            

# Compute 3 x 3 covariance C matrix
def covariance(S_hh, S_hv, S_vv):
    # S_hh, S_hv, S_vv = Single-look complex SAR channels
    
    # build scattering vector
    k1 = S_hh
    k2 = np.sqrt(2) * S_hv
    k3 = S_vv
    
    # 3 x 3 covariance matrix
    H, W = S_hh.shape
    C = np.zeros((3, 3, H, W), dtype=np.complex64)

    C[0, 0] = k1 * np.conj(k1) # |S_hh|^2
    C[0, 1] = k1 * np.conj(k2)
    C[0, 2] = k1 * np.conj(k3)
    C[1, 0] = k2 * np.conj(k1)
    C[1, 1] = k2 * np.conj(k2) # |S_hv|^2
    C[1, 2] = k2 * np.conj(k3)
    C[2, 0] = k3 * np.conj(k1)
    C[2, 1] = k3 * np.conj(k2)
    C[2, 2] = k3 * np.conj(k3) # |S_vv|^2
    
    return C

# Compute 3 x 3 coherency T matrix
def coherency(S_hh, S_hv, S_vv):
    # S_hh, S_hv, S_vv = Single-look complex SAR channels
    
    # build Pauli scattering vector
    k1 = (S_hh + S_vv) / np.sqrt(2)
    k2 = (S_hh - S_vv) / np.sqrt(2)
    k3 = np.sqrt(2) * S_hv
    
    # 3 x 3 coherency matrix
    H, W = S_hh.shape
    T = np.zeros((3, 3, H, W), dtype=np.complex64)

    T[0, 0] = k1 * np.conj(k1)
    T[0, 1] = k1 * np.conj(k2)
    T[0, 2] = k1 * np.conj(k3)
    
    T[1, 0] = k2 * np.conj(k1)
    T[1, 1] = k2 * np.conj(k2)
    T[1, 2] = k2 * np.conj(k3)
    
    T[2, 0] = k3 * np.conj(k1)
    T[2, 1] = k3 * np.conj(k2)
    T[2, 2] = k3 * np.conj(k3)
    
    return T

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
    #ax.set_xlim(feo_ext[0], feo_ext[1])
    #ax.set_ylim(feo_ext[2], feo_ext[3])
    plt.savefig(out_folder+fname, dpi = 300, bbox_inches='tight')
    plt.close()
    
def save_gcp_tiff(hh_slc, N_az, N_rg, outname, warped_out):
    src = gdal.Open(folder + hh_slc) # read GCPs from original SLC file
    gcps = src.GetGCPs()
    gcp_proj = src.GetGCPProjection()
    
    scaled_gcps = [] # scale GCPs to account for multilooking
    for gcp in gcps:
        new_gcp = gdal.GCP(
            gcp.GCPX, gcp.GCPY, gcp.GCPZ,
            gcp.GCPPixel / N_rg,
            gcp.GCPLine / N_az)
        scaled_gcps.append(new_gcp)
    
    dst = gdal.Open(outname, gdal.GA_Update)
    dst.SetGCPs(scaled_gcps, gcp_proj)
    dst = None
    src = None
    gdal.Warp(warped_out, outname, format='GTiff', tps=True, resampleAlg='bilinear', dstSRS=gcp_proj)
    
    
def main():
    
    # to compute and store C and T matrices
    hh_slc = [file for file in files if file.endswith('.tif') and 'hh' in file][0] # hh file
    hv_slc = [file for file in files if file.endswith('.tif') and 'hv' in file][0]
    vv_slc = [file for file in files if file.endswith('.tif') and 'vv' in file][0]
    
    # meta for saving outputs
    """hh_gri = [file for file in all_files if file.endswith('.tif') and 'gri' and 'hh' in file][0] # gri file
    #print_meta(hh_gri)
    meta = get_meta(hh_slc) # meta for saving derived products """

    S_hh = RadCal_SLC(hh_slc, sli_xml) # linear sigma0 ; radimetrically calibrated complex DN from SLC image    
    S_hv = RadCal_SLC(hv_slc, sli_xml)
    S_vv = RadCal_SLC(vv_slc, sli_xml)
    
    """Cmat = covariance(S_hh, S_hv, S_vv)
    print(f"Covariance C Matrix: {np.min(Cmat):.3f} to {np.max(Cmat):.3f}")"""
    
    # T matrix computation ; every cell in SLC gets a 3 x 3 T matrix ; requires complex scattering matrix (SLC) input
    Tmat = coherency(S_hh, S_hv, S_vv)
    print(f"Coherency T Matrix: {np.nanmin(Tmat):.3f} to {np.nanmax(Tmat):.3f}")
    print(Tmat.shape)
    
    # Multilooked 
    #N_az=19
    #N_rg=1
    N_rg, N_az = getLookFactor(gri_xml)
    N_rg, N_az = int(N_rg), int(N_az)
    Tml = multilook(Tmat, N_az, N_rg) # keep N_ra = 1
    print(f"Multilooked Coherency T Matrix: {np.nanmin(Tml):.3f} to {np.nanmax(Tml):.3f}")
    print(Tml.shape)
    
    plot_img(Tml[0,0], hh_slc[:-17]+"_Tml11.png", "T 11 [dB]")
    plot_img(Tml[1,1], hh_slc[:-17]+"_Tml22.png", "T 22 [dB]")
    plot_img(Tml[2,2], hh_slc[:-17]+"_Tml33.png", "T 33 [dB]")

    # Save as GeoTIFF (only diagonal)
    outname = out_folder+hh_slc[:-17]+"_Tml.tif"
    """meta2 = meta.copy()
    meta2.update(count = 3, dtype = "float32", height=Tml.shape[2], width=Tml.shape[3])"""
    meta2 = {"driver": "GTiff", "height": Tml.shape[2], "width": Tml.shape[3], "count": 3, "dtype": "float32"}
    with rio.open(outname, 'w', **meta2) as dst:
        dst.write(Tml[0,0].real.astype(np.float32), 1)
        dst.write(Tml[1,1].real.astype(np.float32), 2)
        dst.write(Tml[2,2].real.astype(np.float32), 3)
    
    warped_out = out_folder+hh_slc[:-17]+"_Tml_warped.tif"
    save_gcp_tiff(hh_slc, N_az, N_rg, outname, warped_out) # add GCP-based projection similar to original DFSAR files

main()