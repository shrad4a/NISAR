\# NISAR S-Band Processing and Feature Extraction



\## 1. Overview



This directory contains the NISAR S-band GCOV processing outputs, numerical feature layers, acquisition metadata, and validation deliverables.



Three S-band acquisitions are processed:



\- 2026-08-19

\- 2026-09-24

\- 2026-10-18



The datasets contain HH and HV polarization information and the HHHV complex cross-covariance term.



\## 2. Source Data



Source products are NISAR S-band GCOV HDF5 products.



\*\*HDF5 covariance terms:\*\*

\- HHHH: HH-HH covariance

\- HVHV: HV-HV covariance

\- HHHV: HH-HV cross-covariance



\*\*Polarizations:\*\* HH and HV



\*\*Product characteristics:\*\*

\- CRS: EPSG:32645 (WGS 84 / UTM Zone 45N)

\- Spatial resolution: 10 m

\- Native grid dimensions: 34200 × 33120

\- AOI dimensions: 2126 × 2006 pixels

\- Radiometric terrain correction: Applied

\- Noise correction: Not applied

\- Polarimetric symmetrization: Applied

\- Full covariance: Not provided



Raw HDF5 files are retained locally and are not included in the GitHub repository.



\## 3. Numerical Feature Extraction



\### 3.1 HH and HV Gamma0 Features



Four numerical feature layers are generated for each acquisition:



| Feature | Description |

|---|---|

| HHHH\_gamma0\_linear.tif | HH Gamma0 in linear scale |

| HHHH\_gamma0\_db.tif | HH Gamma0 in decibel scale |

| HVHV\_gamma0\_linear.tif | HV Gamma0 in linear scale |

| HVHV\_gamma0\_db.tif | HV Gamma0 in decibel scale |



The decibel conversion is:



dB = 10 × log10(Gamma0)



\### 3.2 HHHV Cross-Covariance Features



The HHHV dataset is complex-valued (complex64). Four representations are extracted:



| Feature | Description |

|---|---|

| HHHV\_real.tif | Real component |

| HHHV\_imaginary.tif | Imaginary component |

| HHHV\_magnitude.tif | Complex magnitude |

| HHHV\_phase\_radians.tif | Complex phase in radians |



The derived representations are:



\- Magnitude = abs(HHHV)

\- Phase = angle(HHHV)



The real and imaginary components preserve the information needed to reconstruct the original complex values.



HHHV is a cross-covariance term, not an independent intensity channel. The diagonal covariance dB conversion is therefore not applied to HHHV.



\## 4. Quality Filtering



Feature extraction uses the following quality conditions:



\- Valid-data mask: mask = 1

\- Accepted input exception flags: 0 and 64

\- Non-finite values are excluded.

\- Invalid output pixels are represented using NoData = -9999.



The accepted exception flag 64 is retained rather than automatically discarded.



\## 5. Spatial Alignment



The AOI extraction window is:



\- Row offset: 7332

\- Column offset: 19359

\- Height: 2006 pixels

\- Width: 2126 pixels



The AOI grid uses EPSG:32645 with 10 m pixel spacing.



Spatial alignment was verified by comparing the original HDF5 HHHH, HVHV and RTC factor datasets against their AOI rasters.



All 4,264,756 pixels matched exactly for each compared dataset and acquisition.



\## 6. HHHV Validation Results



All 12 exported HHHV GeoTIFFs were validated against the original HDF5 complex dataset.



| Acquisition | Valid pixels | NoData pixels |

|---|---:|---:|

| 20260819 | 4,264,755 | 1 |

| 20260924 | 4,264,755 | 1 |

| 20261018 | 4,264,756 | 0 |



Validation checks included:



\- CRS, dimensions and geotransform.

\- Float32 output datatype.

\- NoData consistency.

\- Numerical comparison against the original HDF5 dataset.

\- Magnitude and phase calculations.



\*\*Maximum numerical error: 0.0 for all 12 output layers.\*\*



All HHHV feature validation tests passed.



\## 7. Directory Structure



```text

S\_BAND/

├── extracted\_features/

│   ├── gamma0/

│   │   ├── 20260819/

│   │   ├── 20260924/

│   │   └── 20261018/

│   └── hhhv/

│       ├── 20260819/

│       ├── 20260924/

│       └── 20261018/

├── validation/

├── scripts/

└── README.md

```



The directory structure above describes the principal deliverable groups; additional processing and validation files may also be present.



\## 8. Processing Notes



\- Gamma0 and Sigma0 are distinct radiometric quantities.

\- The RTC Gamma-to-Sigma factor is available separately in the source product.

\- Sigma0 can be derived as Gamma0 multiplied by the RTC Gamma-to-Sigma factor.

\- The RTC factor must not be applied twice.

\- The source products have noise correction disabled, which should be considered when interpreting low-backscatter pixels.

\- HHHV magnitude and phase should be interpreted as complex cross-covariance features, not as direct water indicators.



\## 9. Current Deliverable Status



\- Three S-band acquisitions inspected.

\- HH/HV Gamma0 numerical features extracted.

\- HHHV real, imaginary, magnitude and phase features extracted.

\- AOI spatial alignment verified.

\- HHHV output validation completed.

\- Water classification and subsequent analytical stages are maintained as separate processing tasks.

