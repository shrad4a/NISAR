
import h5py
from pathlib import Path

folder = Path(r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5")

files = list(folder.glob("*.h5"))

keys = [
    "/science/SSAR/GCOV/grids/frequencyA/listOfCovarianceTerms",
    "/science/SSAR/GCOV/grids/frequencyA/listOfPolarizations",
    "/science/SSAR/GCOV/grids/frequencyA/xCoordinateSpacing",
    "/science/SSAR/GCOV/grids/frequencyA/yCoordinateSpacing",
    "/science/SSAR/GCOV/grids/frequencyA/xCoordinates",
    "/science/SSAR/GCOV/grids/frequencyA/yCoordinates",
    "/science/SSAR/GCOV/metadata/ceosAnalysisReadyData/outputBackscatterDecibelConversionFormula",
    "/science/SSAR/GCOV/metadata/processingInformation/parameters/radiometricTerrainCorrectionApplied",
    "/science/SSAR/GCOV/metadata/processingInformation/parameters/noiseCorrectionApplied",
    "/science/SSAR/GCOV/metadata/processingInformation/parameters/isFullCovariance",
    "/science/SSAR/GCOV/metadata/processingInformation/parameters/polarimetricSymmetrizationApplied",
]

for file in sorted(files):
    print("\n" + "=" * 80)
    print("ACQUISITION:", file.name)

    with h5py.File(file, "r") as h5:
        for key in keys:
            if key not in h5:
                print("\nMISSING:", key)
                continue

            ds = h5[key]
            value = ds[()]

            if key.endswith("xCoordinates") or key.endswith("yCoordinates"):
                value = {
                    "first": float(value[0]),
                    "second": float(value[1]),
                    "last": float(value[-1]),
                    "count": len(value)
                }
            elif hasattr(value, "tolist"):
                value = value.tolist()

            print("\n", key)
            print("dtype:", ds.dtype)
            print("value:", value)
