# NISAR-Based Surface Water Mapping and L+S Dual-Frequency Fusion

## 1. Project Overview

This project investigates the use of NASA-ISRO Synthetic Aperture Radar (NISAR) observations for surface-water mapping and temporal analysis over a selected area of the Ganga alluvial plain in Bihar, India.

The workflow uses NISAR Level-2 Geocoded, terrain-corrected covariance (GCOV) products from both L-band and S-band acquisitions. HH and HV polarization data are processed to generate numerical radar features and preliminary water candidates. The project also investigates whether combining L-band and S-band information can improve water detection compared with using either frequency independently.

The work is organized into three major components:

1. **L-band processing:** Extraction of HH/HV backscatter features, radiometric analysis and provisional water-candidate classification.
2. **S-band processing:** Extraction and validation of HH/HV Gamma0 features and HHHV complex cross-covariance features.
3. **L+S dual-frequency fusion:** Development and evaluation of methods to combine information from both frequency bands for improved water mapping.

The project currently focuses on establishing reliable, reproducible inputs and processing workflows before conducting final fusion and quantitative water-map validation.

---

## 2. Study Area

The study area is located in the Ganga alluvial plain of Bihar, India.

| Parameter | Value |
|---|---|
| Region | Ganga alluvial plain, Bihar, India |
| AOI longitude | 86.3946°E – 86.6054°E |
| AOI latitude | 25.4098°N – 25.5902°N |
| Approximate AOI area | 425.34 km² |
| Projected CRS | EPSG:32645 |
| Raster dimensions | 2,006 rows × 2,126 columns |
| Pixel spacing | 10 m |

The area of interest (AOI) represents a selected portion of the larger NISAR scene footprint. All AOI-derived outputs should therefore be interpreted as regional subsets rather than full-scene products.

The 10 m value represents pixel spacing and should not be interpreted as the sensor's effective spatial resolution.

---

## 3. Input Data

Three NISAR acquisitions are currently used across the processing workflows.

| Acquisition date | L-band | S-band |
|---|---|---|
| 19 August 2026 | Available | Available |
| 24 September 2026 | Available | Available |
| 18 October 2026 | Available | Available |

### 3.1 L-band Data

The L-band workflow uses NISAR Level-2 GCOV Frequency A products.

The current working dataset contains:

- Ascending orbit observations.
- Track 098 / Frame 014.
- Processing mode: PR DHDH.
- HH and HV polarization data.
- 10 m pixel spacing.

The L-band products are used for backscatter extraction, radiometric statistics and preliminary water-candidate classification.

### 3.2 S-band Data

The S-band workflow uses NISAR S-band GCOV HDF5 products containing:

| Dataset | Description |
|---|---|
| HHHH | HH-HH diagonal covariance |
| HVHV | HV-HV diagonal covariance |
| HHHV | HH-HV complex cross-covariance |

The S-band processing uses HH and HV polarization information, together with the HHHV complex cross-covariance term.

Key product characteristics documented for the current S-band dataset:

- CRS: EPSG:32645.
- Pixel spacing: 10 m.
- Radiometric terrain correction: Applied.
- Noise correction: Not applied.
- Polarimetric symmetrization: Applied.
- Full covariance: Not provided.

The acquisition geometry and processing metadata of each frequency should be checked independently. L-band and S-band observations must not be assumed to have interchangeable acquisition or quality characteristics.

### 3.3 Source Data Management

Raw HDF5 products are retained locally and are not included in the GitHub repository because of their large file sizes.

Derived rasters, processing scripts, metadata inventories and validation reports are organized separately by frequency and processing stage.

---

## 4. Overall Processing Workflow

The project follows a staged workflow:

1. Inspect source GCOV products and metadata.
2. Extract the required polarization and covariance datasets.
3. Define and extract the common area of interest.
4. Generate numerical radar features.
5. Perform radiometric and spatial quality checks.
6. Generate preliminary water-candidate masks.
7. Prepare independent optical reference data.
8. Develop L-band and S-band fusion methods.
9. Evaluate classification performance.
10. Investigate temporal changes in surface-water extent.

L-band and S-band are first processed independently. Their outputs are then prepared for comparison and potential integration through the separate L+S fusion workflow.

---

## 5. L-band Processing

### 5.1 Feature Extraction

The L-band workflow extracts HH and HV backscatter layers from the GCOV products and crops them to the selected AOI.

The resulting rasters use EPSG:32645 and 10 m pixel spacing.

### 5.2 Radiometric Analysis

For each acquisition, descriptive statistics and backscatter distributions are calculated for HH and HV.

Selected median backscatter values from the current pilot are:

| Acquisition date | HH median (dB) | HV median (dB) |
|---|---:|---:|
| 19 August 2026 | -10.299 | -18.732 |
| 24 September 2026 | -10.360 | -18.322 |
| 18 October 2026 | -10.355 | -18.735 |

HH-HV contrast and correlation are also examined to understand the variation between polarizations and acquisitions.

These values represent scene/AOI statistics, not water-specific backscatter measurements. Differences between acquisition dates cannot automatically be interpreted as hydrological changes.

### 5.3 Provisional Water-Candidate Classification

A preliminary baseline classifier has been developed using:

- Scene-specific Otsu thresholds for HH and HV.
- Agreement between dark-pixel candidates from both polarizations.
- Local texture information using a 5 × 5 neighbourhood.
- A 3 × 3 neighbourhood-consistency check.
- A separate uncertain class to avoid forcing every pixel into water or non-water.

The classification uses the following output labels:

| Class value | Meaning |
|---|---|
| 0 | Non-water-like |
| 1 | High-confidence water candidate |
| 2 | Uncertain |
| 255 | Invalid / NoData, where applicable |

The provisional high-confidence candidate counts are:

| Acquisition date | Candidate pixels | Approx. candidate area | Share of AOI |
|---|---:|---:|---:|
| 19 August 2026 | 670,676 | 67.07 km² | 15.73% |
| 24 September 2026 | 852,446 | 85.24 km² | 19.99% |
| 18 October 2026 | 716,609 | 71.66 km² | 16.80% |

These values describe the output of the preliminary classifier only. They are not validated surface-water area estimates and must not be presented as proof of a flood event.

### 5.4 Optical Reference Preparation

An independent Sentinel-2 optical reference dataset has been prepared for the 19 August 2026 pilot.

The selected optical observation was acquired at approximately 05:01 UTC, while the corresponding NISAR acquisition was approximately 23:38 UTC on the same date. The optical observation therefore preceded the radar observation by approximately 18 hours and 37 minutes.

Visible, NIR and SWIR bands, together with the Scene Classification Layer (SCL), are available for reference preparation.

Cloud, cloud-shadow, invalid and ambiguous pixels must be screened before optical labels are used for comparison.

The optical reference is supporting evidence rather than error-free ground truth. Independent reference data and quantitative validation remain pending for the other acquisition dates.

---

## 6. S-band Processing

### 6.1 HH and HV Gamma0 Feature Extraction

Four numerical feature layers are extracted for each acquisition:

| Feature | Description |
|---|---|
| HHHH_gamma0_linear.tif | HH Gamma0 in linear scale |
| HHHH_gamma0_db.tif | HH Gamma0 in decibel scale |
| HVHV_gamma0_linear.tif | HV Gamma0 in linear scale |
| HVHV_gamma0_db.tif | HV Gamma0 in decibel scale |

The conversion from linear Gamma0 to decibel scale is:

\[
\mathrm{Gamma0}_{dB}=10\log_{10}(\mathrm{Gamma0}_{linear})
\]

Gamma0 and Sigma0 are distinct radiometric quantities. The RTC Gamma-to-Sigma factor is available separately in the source product, and Sigma0 can be derived by multiplying Gamma0 by this factor. The factor must not be applied more than once.

### 6.2 HHHV Complex Cross-Covariance Extraction

The HHHV dataset is complex-valued (complex64). Four representations are extracted for each acquisition:

| Feature | Description |
|---|---|
| HHHV_real.tif | Real component |
| HHHV_imaginary.tif | Imaginary component |
| HHHV_magnitude.tif | Complex magnitude |
| HHHV_phase_radians.tif | Complex phase in radians |

The derived quantities are:

\[
\mathrm{Magnitude}=|HHHV|
\]

\[
\mathrm{Phase}=\angle(HHHV)
\]

The real and imaginary components retain the information needed to reconstruct the original complex values.

HHHV represents a cross-covariance term, not an independent intensity channel. Therefore, the diagonal covariance decibel conversion is not applied to HHHV.

### 6.3 Quality Filtering

The S-band feature extraction workflow applies the following conditions:

- Valid-data mask: `mask = 1`.
- Accepted input exception flags: `0` and `64`.
- Non-finite values are excluded.
- Invalid output pixels are represented using `NoData = -9999`.

The accepted exception flag 64 is retained rather than automatically discarded.

### 6.4 Spatial Alignment

The S-band AOI extraction window uses:

| Parameter | Value |
|---|---:|
| Row offset | 7332 |
| Column offset | 19359 |
| Height | 2006 pixels |
| Width | 2126 pixels |
| CRS | EPSG:32645 |
| Pixel spacing | 10 m |

Spatial alignment was checked by comparing the original HDF5 HHHH, HVHV and RTC factor datasets with their corresponding AOI rasters.

All 4,264,756 pixels matched exactly for each compared dataset and acquisition.

### 6.5 HHHV Validation

All 12 exported HHHV GeoTIFF layers across the three acquisitions were validated against their original HDF5 complex datasets.

| Acquisition date | Valid pixels | NoData pixels |
|---|---:|---:|
| 19 August 2026 | 4,264,755 | 1 |
| 24 September 2026 | 4,264,755 | 1 |
| 18 October 2026 | 4,264,756 | 0 |

Validation included CRS, dimensions, geotransform, Float32 datatype, NoData consistency, numerical comparison, magnitude calculation and phase calculation.

**Maximum numerical error: 0.0 across all 12 exported HHHV layers.**

All HHHV feature validation tests passed.

---

## 7. L+S Dual-Frequency Fusion

### 7.1 Objective

The primary objective of the fusion component is to investigate whether combining NISAR L-band and S-band observations improves surface-water detection.

The two frequency bands provide separate radar observations and feature sets. Their complementary information will be investigated rather than assuming that either frequency is inherently more accurate.

### 7.2 Planned Experiments

The fusion workflow is organized into the following experiments:

1. **Repository and data compatibility audit:** Check source data, metadata, acquisition timing, quality flags and feature availability.
2. **Spatial alignment:** Ensure compatible CRS, grid dimensions, pixel spacing and AOI coverage.
3. **Decision-level fusion:** Combine independently generated L-band and S-band water-candidate classifications.
4. **Feature-level fusion:** Combine numerical L-band and S-band features before classification.
5. **Confidence-aware fusion:** Incorporate confidence and uncertainty information from the individual frequency classifications.
6. **Validation and comparison:** Compare L-band, S-band and fused results using independent reference data and appropriate metrics.
7. **Temporal water change detection:** Investigate changes between the three acquisition dates using the resulting water-probability maps and masks.

### 7.3 Expected Outputs

The fusion component is intended to produce:

- Fused water masks.
- Water probability maps.
- Validation metrics.
- L-band versus S-band versus L+S comparison results.
- Temporal surface-water change maps.

**Important:** These are planned fusion deliverables. The availability of separate L-band and S-band processing outputs does not mean that dual-frequency fusion has already been completed or validated.

---

## 8. Quality Control and Limitations

Several considerations are important when interpreting the current outputs.

### 8.1 Provisional Classification

The current L-band water classifier is a heuristic baseline. Its thresholds and candidate areas have not been independently validated across all three dates.

Dark SAR responses are not unique to open water. Smooth soil, radar shadow, urban surfaces and other land-cover conditions can also produce low backscatter.

### 8.2 Quality-Flag Differences

The S-band feature extraction explicitly applies the valid-data mask and accepted exception flags 0 and 64.

The initial L-band radiometric validation used finite, positive backscatter values and available HH/HV/frequency masks, but did not include the `inputDataExceptionMask`.

Consequently, the initial L-band valid-pixel percentages do not represent a complete product-quality audit. Quality filtering should be made consistent and documented before cross-frequency comparisons.

### 8.3 Radiometric Considerations

The S-band source products have noise correction disabled. This should be considered when interpreting low-backscatter pixels.

Gamma0 and Sigma0 must be treated as separate radiometric quantities. The RTC Gamma-to-Sigma factor must not be applied twice.

### 8.4 Cross-Frequency Compatibility

L-band and S-band observations should be checked independently for acquisition geometry, timing, masks, radiometric characteristics and spatial coverage.

A shared CRS and pixel spacing alone do not establish complete data compatibility.

### 8.5 Reference Validation

The Sentinel-2 optical reference has currently been prepared for the 19 August 2026 pilot. Independent reference preparation and quantitative validation remain necessary for the other dates.

No final water-extent, flood-event or causal claims should be made from provisional candidate masks alone.

### 8.6 Temporal Interpretation

Backscatter and classification differences across acquisitions may be influenced by acquisition geometry, hydrological conditions, vegetation, soil moisture and processing metadata. Temporal differences should therefore be interpreted cautiously.

---

## 9. Repository Structure

The project is organized into three main processing directories:

```text
NISAR_WATER_MAPPING/
│
├── L_BAND/
│   ├── METADATA/
│   ├── RAW_HDF5/
│   ├── PROCESSED/
│   │   └── AOI_CROPS/
│   │       ├── 20260819/
│   │       ├── 20260924/
│   │       └── 20261018/
│   ├── WATER_MASKS/
│   │   └── PROVISIONAL_BASELINE_V1/
│   └── RESULTS/
│
├── S_BAND/
│   ├── extracted_features/
│   │   ├── gamma0/
│   │   │   ├── 20260819/
│   │   │   ├── 20260924/
│   │   │   └── 20261018/
│   │   └── hhhv/
│   │       ├── 20260819/
│   │       ├── 20260924/
│   │       └── 20261018/
│   ├── validation/
│   ├── scripts/
│   └── README.md
│
├── L_PLUS_S/
│   └── README.md
│
└── README.md
```

This is the principal logical structure. Additional scripts, intermediate rasters, reports and metadata files may be present in the working repository. Exact filenames and paths should be checked against the current repository contents before executing processing scripts.

Large raw HDF5 files and some derived rasters may be stored outside GitHub.

---

## 10. Current Project Status

The following table summarizes the overall progress across the three components.

| Component | Current status |
|---|---|
| Three L-band acquisitions identified | Complete |
| L-band HDF5 metadata inventory | Available |
| L-band AOI HH/HV backscatter extraction | Complete |
| L-band preliminary radiometric summaries | Complete, with QC caveat |
| L-band provisional water-candidate masks | Generated |
| Sentinel-2 reference preparation for 19 August | Pilot prepared |
| Independent L-band quantitative validation | Pending for all dates |
| Three S-band acquisitions inspected | Complete |
| S-band HH/HV Gamma0 feature extraction | Complete |
| S-band HHHV feature extraction | Complete |
| S-band AOI spatial alignment verification | Complete |
| S-band HHHV numerical validation | Complete; maximum error 0.0 |
| L-band/S-band compatibility audit | To be completed |
| Cross-frequency feature alignment | To be completed |
| Decision-level fusion | Planned |
| Feature-level fusion | Planned |
| Confidence-aware fusion | Planned |
| Comparative quantitative validation | Pending |
| Temporal water-change analysis | Planned |

---

## 11. Reproducibility and Documentation

Every processing run should record:

- Complete source HDF5 filename and product version / CRID.
- Acquisition date and time.
- Orbit direction, track, frame and processing mode.
- Frequency and polarization layers used.
- AOI bounds, CRS, pixel spacing and output dimensions.
- Quality masks and exception-flag handling.
- Script version, parameters and software dependencies.
- Reference-data source, acquisition time and cloud/shadow screening.
- Output paths and validation statistics.

Date-specific outputs and intermediate files should be retained wherever possible so that each derived feature and classification result can be traced back to its source acquisition.

---

## 12. Project Documentation

The individual processing directories contain more detailed information about each workflow:

- `L_BAND/README.md` — L-band data preparation, backscatter analysis, provisional classification and optical reference preparation.
- `S_BAND/README.md` — S-band Gamma0 and HHHV feature extraction, spatial alignment and numerical validation.
- `L_PLUS_S/README.md` — Dual-frequency fusion objectives, planned experiments and expected outputs.

For product-specific information, refer to the NASA/ASF NISAR documentation, NISAR Level-2 GCOV product documentation and ASF OpenSARlab NISAR GCOV Cookbook.

---

## 13. Overall Project Goal

The project establishes separate and reproducible L-band and S-band processing workflows for NISAR-based surface-water mapping over the Ganga alluvial plain.

The L-band workflow has produced preliminary backscatter features and water-candidate classifications, while the S-band workflow has completed numerical feature extraction and rigorous HHHV validation.

The next major stage is to bring the two frequency datasets into a common analytical framework, evaluate decision-level and feature-level fusion approaches, and compare their performance against independently prepared reference data.

The intended outcome is to determine whether dual-frequency L+S information provides more reliable surface-water detection and temporal analysis than either frequency considered independently.
