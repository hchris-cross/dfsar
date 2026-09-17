#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:15:57 2026

@author: hephzy
"""

from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import rasterio as rio
from rasterio.warp import reproject, Resampling
import pandas as pd
import xml.etree.ElementTree as ET
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

home = '/projects/cos-lab-iganesh3/proj/lunar_dfsar/data/Amundsen/' # CEDAR on Bumi 
#home = r"C:\Users\sschatten3\Documents\Amundsen" # win local ; no \ at the end of string

### READ SLC TIF AS NUMPY ARRAY
def readSLC(folder, file):
    print("Reading " + file+"----------", flush=True)
    with rio.open(os.path.join(folder, file)) as src:
        meta = src.meta # Get the metadata of the input file to preserve it for the output
        
        # Read I and Q images
        I = src.read(1).astype(np.float32)  # Real (I) band
        Q = src.read(2).astype(np.float32) # imaginary (Q) band
        I[I == 0] = np.nan
        Q[Q == 0] = np.nan
        
        complex_DN = I + 1j * Q # complex pixel values: DN_complex = I + Qi
        #DN = np.abs(I + 1j * Q) # Magnitude of DN
        #phase = np.angle(complex_DN) # [rad] ; -180 to 180
        
        #print(f"DN: {np.nanmin(complex_DN):.3e} to {np.nanmax(complex_DN):.3e}", flush=True)
        #print(f"Phase: {np.nanmin(phase):.3f} to {np.nanmax(phase):.3f} rad")
        
        return complex_DN

### PRINT TRANSFORM META INFO OF ORIGINAL TIFF FILES
def print_meta(folder, file):
    with rio.open(os.path.join(folder, file)) as src: # rio.open(folder+file)
        print(f"{file}: Shape = {src.height} x {src.width} ; bands = {src.count}")
        
### GET TRANSFORM META INFO OF ORIGINAL TIFF FILES
def get_meta(folder, file):
    with rio.open(os.path.join(folder, file)) as src:
        meta = src.meta
        return meta
    
### GET RADIOMETRIC CALIBRATION COEFFICIENT FROM SLI META
def getCalibCoeff(folder, xml_file):
    # parse calibration constant from XML
    tree = ET.parse(os.path.join(folder, xml_file)) # Parse sli XML file ; ET.parse(folder + xml_file)
    root = tree.getroot()
    K_dB = np.float32(root.find('.//{*}calibration_constant').text) # Extract the calibration constant value
    
    return K_dB

# Radiometrically calibrate SLC images using XML constants
def RadCal_SLC(file, folder, xml_file):
    # parse calibration constant from XML
    tree = ET.parse(os.path.join(folder, xml_file)) # Parse sli XML file ; ET.parse(folder + xml_file)
    root = tree.getroot()
    K_dB = np.float32(root.find('.//{*}calibration_constant').text) # Extract the calibration constant value
    NES0 = -27.9 # Postlaunch NESZ Value for signal power normalized to 100 km altitude and at 30° angle of incidence (B2021)
    #print(f"Calibration Constant: {K_dB}")
    
    complex_DN = readSLC(folder, file) # complex DN from SLC image
    
    # Radiometric calibration
    sigma0 = complex_DN / np.sqrt(10**(K_dB/10)) # linear sigma0  or S = complex_DN / np.sqrt(10**(K_dB/10))
    #sigma0_dB = 10 * np.log10(np.abs(sigma0)**2) # [dB] ;  real valued power
    sigma0_dB = 20*np.log10(np.abs(complex_DN)) # - K_dB
    print(f"sigma0 [dB]: {np.nanmin(sigma0_dB):.3f} to {np.nanmax(sigma0_dB):.3f}", flush=True)
        
    return sigma0 #, sigma0_dB

# Compute 3 x 3 coherency T matrix
def coherency(S_hh, S_hv, S_vv):
    # S_hh, S_hv, S_vv = Single look complex SAR channels
    
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

# Compute only diagonal elements of coherency T matrix
def coherency_diag(S_hh, S_hv, S_vv):
    T11 = 0.5 * np.abs(S_hh + S_vv)**2
    T22 = 0.5 * np.abs(S_hh - S_vv)**2
    T33 = 2 * np.abs(S_hv)**2
    
    return T11, T22, T33

# GET RANGE AND AZIMUTH LOOK FACTORS FROM XML
def getLookFactor(folder, xml_file):
    tree = ET.parse(os.path.join(folder, xml_file)) # Parse the XML file
    root = tree.getroot()
    N_rg = np.float32(root.find('.//{*}range_looks').text) # No. of looks along range
    N_az = np.float32(root.find('.//{*}azimuth_looks').text) # No. of looks along azimuth
    
    print(f"Azimuth looks: {N_az}, range looks: {N_rg}", flush=True)
    return N_rg, N_az

# MULTILOOKING
def multilook(Mq, N_az, N_rg):

    if Mq.ndim == 2:
        H, W = Mq.shape

        H2 = (H // N_az) * N_az
        W2 = (W // N_rg) * N_rg

        Mq = Mq[:H2, :W2]

        Hm = H2 // N_az
        Wm = W2 // N_rg

        return Mq.reshape(Hm, N_az, Wm, N_rg).mean(axis=(1,3))

    elif Mq.ndim == 4:
        _, _, H, W = Mq.shape

        H2 = (H // N_az) * N_az
        W2 = (W // N_rg) * N_rg

        Mq = Mq[:, :, :H2, :W2]

        Hm = H2 // N_az
        Wm = W2 // N_rg

        Ml = np.empty((3,3,Hm,Wm), dtype=Mq.dtype)

        for i in range(3):
            for j in range(3):
                Ml[i,j] = (Mq[i,j].reshape(Hm, N_az, Wm, N_rg).mean(axis=(1,3)))
        return Ml

    else:
        raise ValueError("Input must be 2D or 4D")
        

### PLOT AND SAVE IMAGE ARRAY
def plot_img(Mq, fname, clabel):
    """Mq = np.abs(Mq)
    Mq = np.where(np.isfinite(Mq) & (Mq > 0), Mq, np.nan)
    Mq = 10 * np.log10(Mq + 1e-10) # [dB]"""
    
    Mq = np.where(np.isfinite(Mq) & (Mq > 0), Mq, np.nan)
    Mq = 10*np.log10(np.abs(Mq))
    vmin, vmax = np.nanpercentile(Mq, (2, 98))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(Mq, cmap='viridis', alpha=1, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(clabel)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.savefig(fname, dpi = 300, bbox_inches='tight') # os.path.join(out_folder, fname)
    plt.close()
    
### COMPUTE CIRCULAR POLARIZATION RATIO
def CPR(S_hh, S_hv, S_vv, N_az, N_rg):
    
    # Form covariance terms
    Ihh = np.abs(S_hh)**2
    Ihv = np.abs(S_hv)**2
    Ivv = np.abs(S_vv)**2
    C   = S_hh * np.conjugate(S_vv)
    
    # Multilook covariance terms
    Ihh_ml = multilook(Ihh, N_az, N_rg)
    Ihv_ml = multilook(Ihv, N_az, N_rg)
    Ivv_ml = multilook(Ivv, N_az, N_rg)
    C_ml   = multilook(C,   N_az, N_rg)
    
    sc= 0.25*(Ihh_ml + Ivv_ml + 4*Ihv_ml - 2*np.real(C_ml))
    oc = 0.25*(Ihh_ml + Ivv_ml + 2*np.real(C_ml))
    cpr = sc / np.maximum(oc, 1e-12)
    
    return cpr

### RESAMPLE AND REPROJECT INPUT RASTER 
def resample_raster(src, target):
    out_ras = np.full((target.height, target.width), np.nan, dtype=np.float32)

    reproject(source=src.read(1),
        destination=out_ras,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=target.transform,
        dst_crs=target.crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan)

    return out_ras

### SAVE RASTER AS GEOTIFF
def save_geotiff(data, ref_dataset, out_folder, outfile):
    
    nodata = -9999.0
    data = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    data[data==0] = nodata
    
    with rio.open(ref_dataset) as ref:
        profile = ref.profile.copy()
    #profile = ref_dataset.profile.copy()
    profile.update(dtype=rio.float32, count=1, compress="lzw", nodata=nodata)

    with rio.open(os.path.join(out_folder, outfile), "w", **profile) as dst: # out_folder + outfile
        dst.write(data.astype(np.float32), 1)
    print(f"Saved {outfile}.", flush=True)
    
    
### WRITE FILE PATHS FOR THE ORTHORECTIFICATION TOOL
def wine_path(path):
    return '"' + 'Z:' + path.replace('/', '\\') + '"'
    
### GENERATE COMMAND TO RUN THE ORTHORECTIFICATION TOOL
def ortho_comm(nn, out_folder, folder, PID, gri_xml):

    ## construct command -------
    #tool = r"C:\Users\hchristopher3\Documents\Georef_CH2_DFSAR_SLC_ML_utility_version4_ENVI_DUMP_WINDOWS\georef_sar_ch2dfsar_ml_slc_30112022_ver4_ENVI_OP_WINDOWS.exe"
    #tool = 'WINEDEBUG=-all wine georef_sar_ch2dfsar_ml_slc_30112022_ver4_ENVI_OP_WINDOWS.exe' # Wine on Linux
    tool = 'WINEDEBUG=-all wine /projects/cos-lab-iganesh3/proj/lunar_dfsar/code/Georef_CH2_DFSAR_SLC_ML_utility_version4_ENVI_DUMP_WINDOWS/georef_sar_ch2dfsar_ml_slc_30112022_ver4_ENVI_OP_WINDOWS.exe' # Wine on Linux

    # arg 1: Input_File_To_Geocode  :: Example Test_even.bin  (should be a float image) ; full path ; non-zero values; can't be dB
    #ip_bin_path_1 =  os.path.join(out_folder, PID[:-7]+ f"_sli_{nn}.bin") # win ----------
    ip_bin_path_1 = wine_path(os.path.join(out_folder, PID[:-7] + f"_sli_{nn}.bin"))

    # arg 2: CH2_Product_Path ::  e.g PRODUCT as assumed
    #Pr_path_2 = os.path.join(home, PID) # win ----------
    Pr_path_2 = wine_path(os.path.join(home, PID))

    # arg 3: slc_xml_meta_string(for L/S Band) :: e.g ch2_sar_ncxl_20191022t065522035_d_sli_xx_fp_xx_g26   (Please note it is without .xml extension
    sli_xml_3 = PID[:-7] + '_sli_xx_fp_xx_' + PID[-3:]

    # arg 4, 5: Num_Azimuth_Looks :: No of Azimuth Looks ; Num_Range_Looks :: No of Range Looks
    N_rg, N_az = getLookFactor(folder, gri_xml)
    N_rg, N_az = int(N_rg), int(N_az)
    #N_rg, N_az = int(N_rg), 40

    N_az_4 = str(N_az)
    N_rg_5 = str(N_rg)

    # arg 6: Output_Path :: Path where Output Needs to be Generated
    #op_path_6 = out_folder # win ----------
    op_path_6 = wine_path(out_folder)

    # arg 8: Output_Line_Pixel_Spacing  :: Output_Line_Pixel_Spacing number of users choice  can be provided here.
    # (Default Values : 75MHz Mode 4  :  7.5 MHz Mode  25   3 MHz Mode  60  2 MHz Mode  90 can be provided).
    op_px_spacing_8 = '4' # '4'

    # arg 7: Output_String :: String with which Geocode file needs to be generated
    op_name_7 = f'{nn}_geocoded_{op_px_spacing_8}m'

    ## command ; execute from tool folder
    comm = tool + ' ' + ip_bin_path_1 + ' ' + Pr_path_2 + ' ' + sli_xml_3 + ' ' + N_az_4 + ' ' + N_rg_5 + ' ' + op_path_6 + ' ' + op_name_7 + ' ' + op_px_spacing_8

    print('\n')
    print(comm)
    return comm

### RUN ORTHORECTIFICATION COMMAND IN TERMINAL
def run_comm(comm):
    result = subprocess.run(comm, shell=True)
    if result.returncode != 0:
        print("Orthorectification failed.")
    else:
        print("Orthorectification completed successfully.")
        
### AMPLITUDE TO dB: a = 20 ; POWER TO dB: a = 10
def to_dB(arr, a):
    arr_dB = a * np.log10(np.abs(arr)) # [dB] ;  real valued power
    arr_dB[~np.isfinite(arr_dB)] = np.nan
    
    return arr_dB
        
### APPLY INCIDENCE ANGLE CORRECTION
def inc_angle_corr(PID, out_folder, folder, nn):
    ###### Incidence angle correction
    # geocoded outputs from the orthorectification GEOREF tool are inputs
    # input .bin file
    with rio.open(os.path.join(out_folder, PID[:-7]+f'_sli_xx_fp_xx_{PID[-3:]}', f"{nn}.bin")) as ds:
        arr = ds.read(1)
        print(arr.shape) 
    
        # SRI incidence angle 
        inc_sri_file = os.path.join(folder, PID[:-8] + f'd_sri_in_fp_xx_{PID[-3:]}.tif')
        with rio.open(inc_sri_file) as src:
            # resample SRI inc to match multilooked SLC
            inc = resample_raster(src, ds)
            inc[(inc == -2) | (inc <= 0) | (inc > 80)] = np.nan
    
    # inc correction
    arr_inc = arr * np.cos(np.radians(26))**3 / np.cos(np.radians(inc))**3
    arr_inc[~np.isfinite(arr_inc)] = np.nan
    #arr_inc_dB = 10 * np.log10(np.abs(arr_inc)**2) # [dB] ;  real valued power
    #arr_inc_dB[~np.isfinite(arr_inc_dB)] = np.nan
    
    return arr_inc
    
    
    

### COMPUTE S-MATRIX
def get_S_matrix(PID, out_folder, folder, files, sli_xml, gri_xml, ml=True, ort=True, inc_corr=True, tiff=True):
    
    # read Single Look images
    hh_slc = [file for file in files if file.endswith('.tif') and 'hh' in file][0] # hh file
    hv_slc = [file for file in files if file.endswith('.tif') and 'hv' in file][0]
    vv_slc = [file for file in files if file.endswith('.tif') and 'vv' in file][0]
    
    # Radiometric calibration of S-matrix
    S_hh = RadCal_SLC(hh_slc, folder, sli_xml) # linear sigma0 ; radimetrically calibrated complex DN from SLC image    
    S_hv = RadCal_SLC(hv_slc, folder, sli_xml)
    S_vv = RadCal_SLC(vv_slc, folder, sli_xml)
    
    if ml is True:
        # Multilook -----------
        N_rg, N_az = getLookFactor(folder, gri_xml)
        N_rg, N_az = int(N_rg), int(N_az)
        #N_rg, N_az = int(N_rg), 40
        
        S_hh_ml = multilook(S_hh, N_az, N_rg) # keep N_ra = 1
        print(f"Multilooked S_hh: {np.nanmin(S_hh_ml):.3e} to {np.nanmax(S_hh_ml):.3e}", flush=True)
        
        S_hv_ml = multilook(S_hv, N_az, N_rg) 
        print(f"Multilooked S_hv: {np.nanmin(S_hv_ml):.3e} to {np.nanmax(S_hv_ml):.3e}", flush=True)
        
        S_vv_ml = multilook(S_vv, N_az, N_rg) 
        print(f"Multilooked S_vv: {np.nanmin(S_vv_ml):.3e} to {np.nanmax(S_vv_ml):.3e}", flush=True)
        
    if ort is True:
        # save as .bin for orthorectification -- BEFORE INC NORMALIZATION
        # tool can't take dB input; only positive real values
        np.abs(S_hh_ml).astype(np.float32).tofile(os.path.join(out_folder, PID[:-7]+"_sli_S_hh_ml_abs.bin"))
        np.abs(S_hv_ml).astype(np.float32).tofile(os.path.join(out_folder, PID[:-7]+"_sli_S_hv_ml_abs.bin"))
        np.abs(S_vv_ml).astype(np.float32).tofile(os.path.join(out_folder, PID[:-7]+"_sli_S_vv_ml_abs.bin"))
        
        S_hh_comm = ortho_comm('S_hh_ml_abs', out_folder, folder, PID, gri_xml)
        S_hv_comm = ortho_comm('S_hv_ml_abs', out_folder, folder, PID, gri_xml)
        S_vv_comm = ortho_comm('S_vv_ml_abs', out_folder, folder, PID, gri_xml)
        
        run_comm(S_hh_comm)
        run_comm(S_hv_comm)
        run_comm(S_vv_comm)
            
    if inc_corr is True:
        # Incidence angle correction
        S_hh_inc = inc_angle_corr(PID, out_folder, folder, 'S_hh_ml_abs_geocoded_4m')
        S_hv_inc = inc_angle_corr(PID, out_folder, folder, 'S_hv_ml_abs_geocoded_4m')
        S_vv_inc = inc_angle_corr(PID, out_folder, folder, 'S_vv_ml_abs_geocoded_4m')
        
        S_hh_inc_dB = to_dB(S_hh_inc, 20)
        S_hv_inc_dB = to_dB(S_hv_inc, 20)
        S_vv_inc_dB = to_dB(S_vv_inc, 20)
        
        ref_dataset = os.path.join(out_folder, PID[:-7]+f'_sli_xx_fp_xx_{PID[-3:]}', "S_hh_ml_abs_geocoded_4m.bin")
        save_geotiff(S_hh_inc_dB, ref_dataset, out_folder, PID[:-7]+"_sli_S_hh_inc_dB.tif")
        save_geotiff(S_hv_inc_dB, ref_dataset, out_folder, PID[:-7]+"_sli_S_hv_inc_dB.tif")
        save_geotiff(S_vv_inc_dB, ref_dataset, out_folder, PID[:-7]+"_sli_S_vv_inc_dB.tif")
        
    if tiff is True:
        S_hh_dB = to_dB(S_hh_ml, 20)
        S_hv_dB = to_dB(S_hv_ml, 20)
        S_vv_dB = to_dB(S_vv_ml, 20)
        
        ref_dataset = os.path.join(out_folder, PID[:-7]+f'_sli_xx_fp_xx_{PID[-3:]}', "S_hh_ml_abs_geocoded_4m.bin")
        save_geotiff(S_hh_dB, ref_dataset, out_folder, PID[:-7]+"_sli_S_hh_dB.tif")
        save_geotiff(S_hv_dB, ref_dataset, out_folder, PID[:-7]+"_sli_S_hv_dB.tif")
        save_geotiff(S_vv_dB, ref_dataset, out_folder, PID[:-7]+"_sli_S_vv_dB.tif")
        
    return S_hh, S_hv, S_vv

### GET T-MATRIX
def get_T_matrix(PID, out_folder, folder, files, S_hh, S_hv, S_vv, sli_xml, gri_xml, ml=True, ort=True, inc_corr=True, tiff=True):
    
    # T matrix computation ; every cell in SLC gets a 3 x 3 T matrix ; requires complex scattering matrix (SLC) input
    #Tmat = coherency(S_hh, S_hv, S_vv) # get all elements
    T11, T22, T33 = coherency_diag(S_hh, S_hv, S_vv) # only diagonal
    print(f"Coherency T Matrix: {np.nanmin(T11):.3e} to {np.nanmax(T11):.3e}", flush=True)
    
    if ml is True:
        # Multilook -----------
        N_rg, N_az = getLookFactor(folder, gri_xml)
        N_rg, N_az = int(N_rg), int(N_az)
        #N_rg, N_az = int(N_rg), 40
        
        #Tml = multilook(Tmat, N_az, N_rg) # keep N_ra = 1
        T11_ml = multilook(T11, N_az, N_rg) # keep N_ra = 1
        T22_ml = multilook(T22, N_az, N_rg) # keep N_ra = 1
        T33_ml = multilook(T33, N_az, N_rg) # keep N_ra = 1
        print(f"Multilooked Coherency T11 Matrix: {np.nanmin(T11_ml):.3e} to {np.nanmax(T11_ml):.3e}", flush=True)
        print(f"Multilooked Coherency T22 Matrix: {np.nanmin(T22_ml):.3e} to {np.nanmax(T22_ml):.3e}", flush=True)
        print(f"Multilooked Coherency T33 Matrix: {np.nanmin(T33_ml):.3e} to {np.nanmax(T33_ml):.3e}", flush=True)
        
    if ort is True:
        # save as .bin for orthorectification -- BEFORE INC NORMALIZATION
        # tool can't take dB input; only positive real values
        np.abs(T11_ml).astype(np.float32).tofile(os.path.join(out_folder, PID[:-7]+"_sli_T11_ml_abs.bin")) # Tml[0,0]
        np.abs(T22_ml).astype(np.float32).tofile(os.path.join(out_folder, PID[:-7]+"_sli_T22_ml_abs.bin"))
        np.abs(T33_ml).astype(np.float32).tofile(os.path.join(out_folder, PID[:-7]+"_sli_T33_ml_abs.bin"))
        
        T11_comm = ortho_comm('T11_ml_abs', out_folder, folder, PID, gri_xml)
        T22_comm = ortho_comm('T22_ml_abs', out_folder, folder, PID, gri_xml)
        T33_comm = ortho_comm('T33_ml_abs', out_folder, folder, PID, gri_xml)
        
        run_comm(T11_comm)
        run_comm(T22_comm)
        run_comm(T33_comm)
        
    if inc_corr is True:
        # Incidence angle correction
        T11_inc = inc_angle_corr(PID, out_folder, folder, 'T11_ml_abs_geocoded_4m')
        T22_inc = inc_angle_corr(PID, out_folder, folder, 'T22_ml_abs_geocoded_4m')
        T33_inc = inc_angle_corr(PID, out_folder, folder, 'T33_ml_abs_geocoded_4m')
        
        T11_inc_dB = to_dB(T11_inc, 10)
        T22_inc_dB = to_dB(T22_inc, 10)
        T33_inc_dB = to_dB(T33_inc, 10)
        
        ref_dataset = os.path.join(out_folder, PID[:-7]+f'_sli_xx_fp_xx_{PID[-3:]}', "S_hh_ml_abs_geocoded_4m.bin")
        save_geotiff(T11_inc_dB, ref_dataset, out_folder, PID[:-7]+"_sli_T11_inc_dB.tif")
        save_geotiff(T22_inc_dB, ref_dataset, out_folder, PID[:-7]+"_sli_T22_inc_dB.tif")
        save_geotiff(T33_inc_dB, ref_dataset, out_folder, PID[:-7]+"_sli_T33_inc_dB.tif")
        
    if tiff is True:
        T11_dB = to_dB(T11_ml, 10)
        T22_dB = to_dB(T22_ml, 10)
        T33_dB = to_dB(T33_ml, 10)
        
        ref_dataset = os.path.join(out_folder, PID[:-7]+f'_sli_xx_fp_xx_{PID[-3:]}', "S_hh_ml_abs_geocoded_4m.bin")
        save_geotiff(T11_dB, ref_dataset, out_folder, PID[:-7]+"_sli_T11_dB.tif")
        save_geotiff(T22_dB, ref_dataset, out_folder, PID[:-7]+"_sli_T22_dB.tif")
        save_geotiff(T33_dB, ref_dataset, out_folder, PID[:-7]+"_sli_T33_dB.tif")
        
    return T11, T22, T33

### GET CPR
def get_CPR(PID, out_folder, folder, files, S_hh, S_hv, S_vv, sli_xml, gri_xml, ml=True, ort=True, inc_corr=True, tiff=True):
    N_rg, N_az = getLookFactor(folder, gri_xml)
    N_rg, N_az = int(N_rg), int(N_az)
    
    cpr = CPR(S_hh, S_hv, S_vv, N_az, N_rg) # uses multilooked S elements
    print(f"cpr value, {np.nanmin(cpr):.3e} to {np.nanmax(cpr):.3e}", flush=True)
    
    if ort is True:
        ###### save as .bin for orthorectification -- BEFORE INC NORMALIZATION
        # tool can't take dB input; only positive real values
        np.abs(cpr).astype(np.float32).tofile(os.path.join(out_folder, PID[:-7]+"_sli_cpr.bin"))
        
        cpr_comm = ortho_comm('cpr', out_folder, folder, PID, gri_xml)
        run_comm(cpr_comm)
        
    if inc_corr is True:
        # Incidence angle correction
        cpr_inc = inc_angle_corr(PID, out_folder, folder, 'cpr')
        
        ref_dataset = os.path.join(out_folder, PID[:-7]+f'_sli_xx_fp_xx_{PID[-3:]}', "S_hh_ml_abs_geocoded_4m.bin")
        save_geotiff(cpr_inc, ref_dataset, out_folder, PID[:-7]+"_sli_cpr_inc.tif")
    
    if tiff is True:
        ref_dataset = os.path.join(out_folder, PID[:-7]+f'_sli_xx_fp_xx_{PID[-3:]}', "S_hh_ml_abs_geocoded_4m.bin")
        save_geotiff(cpr, ref_dataset, out_folder, PID[:-7]+"_sli_cpr.tif")
            
    return cpr
    



def main(PID, folder, out_folder, files, sli_xml, gri_xml):
    
    ######### S-matrix
    S_hh, S_hv, S_vv = get_S_matrix(PID, out_folder, folder, files, sli_xml, gri_xml, ml=True, ort=True, inc_corr=True, tiff=True)
    
    #plot_img(multilook((abs(S_hh))**2, N_az, N_rg), os.path.join(out_folder, hh_slc[:-17]+'_S_hh.png'), 'Backscatter [dB]')
    #plot_img(multilook((abs(S_hv))**2, N_az, N_rg), os.path.join(out_folder, hh_slc[:-17]+'_S_hv.png'), 'Backscatter [dB]')
    
    ######## T-matrix
    T11, T22, T33 = get_T_matrix(PID, out_folder, folder, files, S_hh, S_hv, S_vv, sli_xml, gri_xml, ml=True, ort=True, inc_corr=True, tiff=True)
    
    ######### CPR
    cpr = get_CPR(PID, out_folder, folder, files, S_hh, S_hv, S_vv, sli_xml, gri_xml, ml=True, ort=True, inc_corr=True, tiff=True)
    
    
    
    
    
def batchProcess():
    Pr_ID_list = os.listdir(home)
    #Pr_ID_list = ["ch2_sar_ncxl_20191106t114537878_d_fp_d18"] # single file
    
    # Only process DFSAR PIDs
    Pr_ID_list = [PID for PID in Pr_ID_list if PID.startswith("ch2_sar")]

    # Number of PIDs to process simultaneously
    n_workers = 1

    print(f"Found {len(Pr_ID_list)} PIDs")
    print(f"Using {n_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_PID, PID): PID for PID in Pr_ID_list}
        for future in as_completed(futures):
            PID = futures[future]
            try:
                result = future.result()
                print(result)

            except Exception as e:
                print(f"ERROR processing {PID}: {e}")
            
            
def process_PID(PID):

    try:
        folder = os.path.join(home, PID, "data", "calibrated", PID[13:21])
        out_folder = os.path.join(folder, "derived")

        all_files = os.listdir(folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            
        # get geometry file
        #inc_slc_file = home + PID + "/geometry/calibrated/" + PID[13:21] + '/' + PID[:-8] +'g_sli_xx_fp_xx_'+ PID[-3:]+'.csv'
        #inc_df = pd.read_csv(inc_slc_file, delimiter=',') # geom file

        # read only SLC files 'sli'
        patterns = ["sli_xx_fp_vv", "sli_xx_fp_hh", "sli_xx_fp_hv", "sli_xx_fp_vh"]
        files = [file for file in all_files if file.endswith('.tif') and any(pattern in file for pattern in patterns)] # keeping only SLI files

        # xml files
        sli_xml = [file for file in all_files if file.endswith('.xml') and "sli_xx_fp_xx" in file][0]
        gri_xml = [file for file in all_files if file.endswith('.xml') and "gri_xx_fp_xx" in file][0]
        
        #sri_file = os.path.join(folder, PID[:-8] + "d_sri_in_fp_xx_" + PID[-3:] + ".tif")

        # Check if already processed
        """db_files = [file for file in os.listdir(out_folder) if file.endswith('_dB.tif')]
        if db_files:
            return f"Skipping {PID}: dB files already exist"""

        print(f"----------{PID}------------", flush=True)

        # Run existing processing
        main(PID, folder, out_folder, files, sli_xml, gri_xml)

        return f"Finished {PID}"

    except Exception as e:

        return f"ERROR processing {PID}: {e}"



if __name__ == "__main__":
    batchProcess()