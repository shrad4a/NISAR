
import h5py
from pathlib import Path

files = [
    Path(r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5\NISAR_S2_PR_GCOV_028_098_A_014_3700_DHNA_A_20260819T233810_20260819T233847_P00500_P_F_I_001 (1).h5"),
    Path(r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5\NISAR_S2_UR_GCOV_031_098_A_014_3700_DHNA_A_20260924T233810_20260924T233846_P00500_M_F_I_001.h5"),
    Path(r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5\NISAR_S2_PR_GCOV_033_098_A_014_3700_DHNA_A_20261018T233810_20261018T233847_P00500_M_F_I_001 (1).h5"),
]

output = Path("S_BAND_HDF5_AUDIT.txt")

with output.open("w", encoding="utf-8") as report:
    for file in files:
        print(f"\n{'=' * 90}\nFILE: {file.name}", file=report)

        if not file.exists():
            print("ERROR: FILE NOT FOUND", file=report)
            continue

        with h5py.File(file, "r") as h5:
            print("\nROOT ATTRIBUTES:", file=report)
            for key, value in h5.attrs.items():
                print(f"  {key}: {value}", file=report)

            print("\nALL GROUPS AND DATASETS:", file=report)

            def inspect(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"\nGROUP: /{name}", file=report)
                    for key, value in obj.attrs.items():
                        print(f"  ATTR {key}: {value}", file=report)

                elif isinstance(obj, h5py.Dataset):
                    print(f"\nDATASET: /{name}", file=report)
                    print(f"  Shape: {obj.shape}", file=report)
                    print(f"  Dtype: {obj.dtype}", file=report)
                    print(f"  Size: {obj.size}", file=report)
                    for key, value in obj.attrs.items():
                        print(f"  ATTR {key}: {value}", file=report)

            h5.visititems(inspect)

        print(f"\nFINISHED: {file.name}", file=report)

print(f"Audit saved to: {output.resolve()}")
