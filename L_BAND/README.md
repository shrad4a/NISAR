# NISAR L-Band Surface-Water Mapping

## Project overview

This directory contains the L-band processing workflow for surface-water
mapping and temporal analysis over a selected part of the Ganga alluvial
plain in Bihar, India. The work uses NISAR Level-2 Geocoded,
terrain-corrected covariance (GCOV) products to derive HH/HV backscatter
features, generate preliminary water candidates, and prepare the L-band
inputs for comparison with S-band and L+S feature fusion.

**Scope:** L-band only. S-band datasets and fusion outputs are
maintained separately in `S_BAND/` and `L_PLUS_S/`.

## Study area

  Parameter               Value
  ----------------------- --------------------------------------------
  Region                  Ganga alluvial plain, Bihar, India
  AOI bounds (WGS84)      86.3946°E--86.6054°E, 25.4098°N--25.5902°N
  Approximate AOI area    425.34 km²
  Projected CRS           EPSG:32645
  AOI raster dimensions   2,006 × 2,126 pixels
  Pixel spacing           10 m

The AOI is used for the current pilot analysis. The source GCOV scenes
cover a larger footprint; AOI crops should not be described as
full-scene outputs.

## Input data

Three NISAR L-band Level-2 GCOV acquisitions are used in the current
pilot:

  ------------------------------------------------------------------------------
  Acquisition   Orbit       Track /     Processing   Mode        Polarizations
  date (UTC)                Frame       type                     
  ------------- ----------- ----------- ------------ ----------- ---------------
  19 Aug 2026   Ascending   098 / 014   PR           DHDH        HH, HV

  24 Sep 2026   Ascending   098 / 014   PR           DHDH        HH, HV

  18 Oct 2026   Ascending   098 / 014   PR           DHDH        HH, HV
  ------------------------------------------------------------------------------

The products are Frequency A, 10 m pixel-spacing products in the current
working dataset. Pixel spacing is not the same as the sensor's effective
spatial resolution. Always retain the complete product filename and
metadata when documenting a run.

NISAR GCOV products contain geocoded, terrain-corrected polarimetric
covariance information. Refer to the [ASF NISAR GCOV
documentation](https://github.com/ua-asf/nisar-docs/blob/main/data-products/level-2/gcov.md)
and [NISAR mission documentation](https://github.com/ua-asf/nisar-docs).

## Directory layout

The local working directory has been organized approximately as follows:

``` text
L_BAND/
├── METADATA/
│   ├── hdf5_dataset_inventory.csv
│   └── hdf5_structure.txt
├── RAW_HDF5/
│   └── NISAR L2 GCOV HDF5 products
├── PROCESSED/
│   └── AOI_CROPS/
│       ├── 20260819/
│       ├── 20260924/
│       └── 20261018/
├── WATER_MASKS/
│   └── PROVISIONAL_BASELINE_V1/
└── RESULTS/
```

Exact filenames and outputs should be checked against the current
repository contents before running scripts. Large source HDF5 files and
derived rasters may be excluded from Git and stored separately.

## Processing workflow

### 1. Product inspection and metadata audit

The HDF5 products are inspected to identify the relevant frequency
group, polarization layers, geolocation information, raster dimensions,
masks, and product metadata. The inventory and structure reports are
retained under `METADATA/`.

### 2. AOI extraction

HH and HV backscatter layers are extracted and cropped to the study-area
bounds. The current AOI crops use EPSG:32645 and 10 m pixel spacing.

### 3. Radiometric statistics

For each date, HH and HV backscatter distributions are summarized using
descriptive statistics and percentiles. HH--HV contrast and correlation
are also calculated to inspect temporal consistency and changes in the
backscatter distributions.

Selected median values from the current pilot:

  Date            HH median (dB)   HV median (dB)
  ------------- ---------------- ----------------
  19 Aug 2026            -10.299          -18.732
  24 Sep 2026            -10.360          -18.322
  18 Oct 2026            -10.355          -18.735

These are scene/AOI summary statistics, not water-specific measurements.
Differences between dates should not automatically be interpreted as
hydrological change.

### 4. Provisional water-candidate classification

A preliminary baseline classifier was generated using:

-   Scene-specific Otsu thresholds for HH and HV backscatter.
-   Agreement between dark-pixel candidates from the two polarizations.
-   Local texture information from a 5 × 5 neighbourhood.
-   A 3 × 3 neighbourhood-consistency check.
-   A separate uncertain class rather than forcing every pixel into
    water or non-water.

The outputs distinguish: - `0`: non-water-like - `1`: high-confidence
water candidate - `2`: uncertain - `255`: invalid / no-data, where
applicable

**These are provisional SAR-derived candidates, not validated final
water masks.** Dark surfaces can include non-water features, and the
classifier requires independent reference validation.

### 5. Reference-data validation

An independent Sentinel-2 optical reference was prepared for the 19
August 2026 pilot. The selected AWS Earth Search item was acquired at
approximately 05:01 UTC on 19 August, while the NISAR acquisition was
approximately 23:38 UTC on the same date. The optical observation
therefore preceded the NISAR acquisition by about 18 hours and 37
minutes.

Optical reference layers include visible/NIR/SWIR bands and the Scene
Classification Layer (SCL). Cloud, cloud-shadow, invalid and ambiguous
pixels must be screened before reference labels are used. Optical
classifications are reference evidence, not error-free ground truth.

Additional dates require appropriately timed independent reference data
before quantitative validation can be claimed.

## Preliminary classification summary

The current provisional baseline produced the following high-confidence
candidate counts:

  -----------------------------------------------------------------------
  Date              High-confidence  Approx. candidate       Share of AOI
                   candidate pixels               area 
  -------------- ------------------ ------------------ ------------------
  19 Aug 2026               670,676          67.07 km²             15.73%

  24 Sep 2026               852,446          85.24 km²             19.99%

  18 Oct 2026               716,609          71.66 km²             16.80%
  -----------------------------------------------------------------------

Area estimates use 10 m × 10 m pixel spacing. These figures describe the
output of the provisional classifier only. They must not be presented as
validated water extent or as proof of a flood event. The uncertain class
is retained and should be reported alongside the candidate area.

## Quality-control notes and limitations

1.  The initial radiometric validation used finite, positive backscatter
    values and the available HH/HV/frequency masks. The
    `inputDataExceptionMask` was not included in that particular
    validation script. Therefore, the reported valid-pixel percentages
    do not represent a complete product-quality audit.
2.  The current water classifier is a heuristic baseline. Its thresholds
    and candidate areas have not yet been independently validated for
    all three dates.
3.  A dark SAR response is not uniquely caused by open water. Smooth
    soil, radar shadow, urban surfaces and other land-cover conditions
    may also appear dark.
4.  Acquisition geometry, hydrological conditions, vegetation, soil
    moisture and processing metadata can influence backscatter. Temporal
    differences should be interpreted cautiously.
5.  The 10 m pixel spacing should not be described as 10 m effective
    spatial resolution.
6.  No causal claim about flooding or its drivers should be made without
    independent supporting evidence.
7.  L-band and S-band are processed separately. Do not assume their
    acquisition geometry, timing, masks or polarimetric measurements are
    interchangeable.

## Reproducibility

For every processing run, record:

-   Full source HDF5 filename and product version / CRID.
-   Acquisition date, orbit direction, track, frame and mode.
-   Frequency and polarization layers used.
-   AOI bounds, CRS, pixel spacing and output dimensions.
-   Mask and quality-flag handling.
-   Script version, parameters and software dependencies.
-   Reference-data source, acquisition time and cloud/shadow screening.
-   Output paths and summary statistics.

Keep intermediate rasters and reports with date-specific names so that
outputs can be traced back to their source acquisition.

## Current status

  -----------------------------------------------------------------------
  Component                           Status
  ----------------------------------- -----------------------------------
  Three L-band GCOV acquisitions      Complete
  identified                          

  HDF5 metadata inventory             Available

  AOI HH/HV backscatter crops         Complete

  Preliminary radiometric summaries   Complete, with QC caveat

  Provisional water-candidate masks   Generated

  Independent optical reference       Pilot prepared for 19 Aug 2026
  preparation                         

  Independent quantitative validation Pending
  for all dates                       

  Final validated L-band water maps   Pending

  L-band vs S-band vs L+S comparison  Maintained in the separate fusion
                                      workflow
  -----------------------------------------------------------------------

## Suggested citation and documentation

-   [NASA/ASF NISAR documentation](https://github.com/ua-asf/nisar-docs)
-   [NISAR Level-2 GCOV product
    documentation](https://github.com/ua-asf/nisar-docs/blob/main/data-products/level-2/gcov.md)
-   [ASF OpenSARlab NISAR GCOV
    Cookbook](https://github.com/ASFOpenSARlab/NISAR_Cookbook)

------------------------------------------------------------------------

**Important:** This README documents the current L-band pilot workflow
and its provisional outputs. Update the status table, paths, script
names and validation results whenever the repository is revised. Do not
describe provisional candidate masks as validated water maps.
