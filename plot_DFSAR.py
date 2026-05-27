#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:17:43 2026

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
    
    ### PLOT
    # phases of each pol channel
    O_hh_ph = np.degrees(np.angle(O_hh))
    O_hv_ph = np.degrees(np.angle(O_hv))
    O_vh_ph = np.degrees(np.angle(O_vh))
    O_vv_ph = np.degrees(np.angle(O_vv))
    
    # observed inter-channel phases
    #phi_co_obs = np.degrees(np.angle(O_hh * np.conj(O_vv)))
    #phi_cx_obs  = np.degrees(np.angle(O_hv * np.conj(O_vh)))
    
    phi_co_obs = O_hh_ph - O_vv_ph
    phi_cx_obs = O_hv_ph - O_vh_ph
    
    # histogram of each channel -------- 
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    bins = 200
    
    # HH
    axes[0,0].hist(O_hh_ph.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[0,0].set_title('HH')
    axes[0,0].legend()
    
    # HV
    axes[0,1].hist(O_hv_ph.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[0,1].set_title('HV')
    axes[0,1].legend()
    
    # VH
    axes[1,0].hist(O_vh_ph.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[1,0].set_title('VH')
    axes[1,0].legend()
    
    # VV
    axes[1,1].hist(O_vv_ph.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[1,1].set_title('VV')
    axes[1,1].legend()
    
    for ax in axes.ravel():
        ax.set_xlabel('Phase (deg)')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(out_folder+'ch2_sar_ncxl_20210824t163808566_d_fp_d18_phase.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # histogram of co pol, cross pol phase diff ------
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), dpi=300)
    bins = 200
    
    # HH
    axes[0].hist(phi_co_obs.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[0].set_title('co-pol')
    axes[0].legend()
    
    # HV
    axes[1].hist(phi_cx_obs.ravel(), bins=bins, alpha=0.5, label='Observed')
    axes[1].set_title('cross-pol')
    axes[1].legend()
    
    for ax in axes.ravel():
        ax.set_xlabel('Phase (deg)')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(out_folder+'ch2_sar_ncxl_20210824t163808566_d_fp_d18_phase_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
main()