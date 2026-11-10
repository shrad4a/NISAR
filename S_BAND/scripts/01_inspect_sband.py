
from pathlib import Path
import h5py
import csv
import json
import numpy as np

# Folder locations
RAW_DIR = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5"
)

OUTPUT_DIR = Path(
      r"C:\Users\HP\Downloads\NISAR\S_BAND\METADATA"
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convert HDF5 attributes into printable values
def safe_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        if value.size <= 50:
            return value.tolist()
        return f"Array: shape={value.shape}"

    if isinstance(value, np.generic):
        return value.item()

    return str(value)


def attributes(obj):
    return {
        str(k): safe_value(v)
        for k, v in obj.attrs.items()
    }


files = sorted(RAW_DIR.rglob("*.h5"))

if not files:
    raise FileNotFoundError(
        f"No HDF5 files found in {RAW_DIR}"
    )

print(f"Found {len(files)} HDF5 files.")

report = []
csv_rows = []

for file in files:

    print("\n" + "=" * 60)
    print("INSPECTING:", file.name)
    print("=" * 60)

    report.append(f"\nFILE: {file.name}")
    report.append(
        f"SIZE: {file.stat().st_size / (1024**3):.3f} GB"
    )

    try:
        if not h5py.is_hdf5(file):
            print("Invalid HDF5 file.")
            continue

        with h5py.File(file, "r") as h5:

            report.append("\nROOT ATTRIBUTES:")
            report.append(
                json.dumps(
                    attributes(h5),
                    indent=2,
                    default=str
                )
            )

            def inspect(name, obj):

                path = "/" + name

                if isinstance(obj, h5py.Group):

                    report.append(f"\n[GROUP] {path}")

                    attrs = attributes(obj)

                    if attrs:
                        report.append(
                            "Attributes: " +
                            json.dumps(attrs, default=str)
                        )

                elif isinstance(obj, h5py.Dataset):

                    attrs = attributes(obj)

                    print(
                        f"{path} | "
                        f"Shape={obj.shape} | "
                        f"Type={obj.dtype}"
                    )

                    report.append(f"\n[DATASET] {path}")
                    report.append(f"Shape: {obj.shape}")
                    report.append(f"Type: {obj.dtype}")
                    report.append(f"Chunks: {obj.chunks}")
                    report.append(
                        f"Compression: {obj.compression}"
                    )
                    report.append(
                        "Attributes: " +
                        json.dumps(attrs, indent=2, default=str)
                    )

                    csv_rows.append({
                        "file": file.name,
                        "dataset_path": path,
                        "shape": str(obj.shape),
                        "dtype": str(obj.dtype),
                        "chunks": str(obj.chunks),
                        "compression": str(obj.compression),
                        "attributes": json.dumps(
                            attrs, default=str
                        )
                    })

            h5.visititems(inspect)

        print("Inspection successful!")

    except Exception as e:
        print("ERROR:", e)
        report.append(f"ERROR: {e}")


# Save text report
txt_file = OUTPUT_DIR / "hdf5_structure.txt"

txt_file.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# Save dataset inventory
csv_file = OUTPUT_DIR / "hdf5_dataset_inventory.csv"

with open(csv_file, "w", newline="", encoding="utf-8") as f:

    columns = [
        "file",
        "dataset_path",
        "shape",
        "dtype",
        "chunks",
        "compression",
        "attributes"
    ]

    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(csv_rows)


print("\nINSPECTION COMPLETE")
print("TXT:", txt_file)
print("CSV:", csv_file)
print("Total datasets:", len(csv_rows))
