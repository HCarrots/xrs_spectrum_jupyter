import h5py
from pathlib import Path
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation



def find_hdf5_file(scan_id, file_path):
    '''
    scanID + file_path search the target files.
    '''
    data_dir = Path(file_path)
    folders = [
        p for p in data_dir.glob(f"{scan_id}_scan_*")
        if p.is_dir()
    ]
    if not folders:
        raise FileNotFoundError(
            f"Cannot find scan folder {scan_id} in {data_dir}"
        )

    if len(folders) > 1:
        raise RuntimeError(
            f"Multiple scan folders found for scan {scan_id}: {folders}"
        )
    scan_folder = folders[0]
    files = list(scan_folder.glob("*.nxs"))
    if not files:
        raise FileNotFoundError(
            f"Cannot find .nxs file in {scan_folder}"
        )

    if len(files) > 1:
        raise RuntimeError(
            f"Multiple .nxs files found in {scan_folder}: {files}"
        )
    return files[0]

def show_hdf5_structure(file_path, show_attrs=False):
    """
    Show the structure of an HDF5 / NeXus file.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 or .nxs file.

    show_attrs : bool, optional
        Whether to display attributes.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    def print_item(name, obj):
        depth = name.count("/")
        indent = "    " * depth

        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 {obj.name}")

        elif isinstance(obj, h5py.Dataset):
            print(
                f"{indent}📄 {obj.name} "
                f"shape={obj.shape}, dtype={obj.dtype}"
            )

        if show_attrs and obj.attrs:
            for key, value in obj.attrs.items():
                print(
                    f"{indent}    └─ @{key}: {value}"
                )

    print(f"HDF5 file: {file_path}")
    print("/")

    with h5py.File(file_path, "r") as f:
        f.visititems(print_item)


def read_hdf5_dataset(file_path, dataset_path, selection=None):
    """
    Read a dataset from an HDF5 / NeXus file as a NumPy array.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.

    dataset_path : str
        Path of the dataset inside the HDF5 file.
        Example:
        "/entry/data/data"

    selection : optional
        NumPy-style selection/slicing.
        If None, read the entire dataset.

    Returns
    -------
    np.ndarray
        Dataset converted to a NumPy array.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, "r") as f:

        if dataset_path not in f:
            raise KeyError(
                f"Dataset not found: {dataset_path}"
            )

        dataset = f[dataset_path]

        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(
                f"{dataset_path} is not a Dataset."
            )

        if selection is None:
            data = dataset[...]
        else:
            data = dataset[selection]

    return np.asarray(data)


def plot_det_image(det_image, log_scale: bool = False, detector_label: str = "D_LAMBDA", figsize=(7,6)):
    """
    Plot summed  detector image (heatmap)
    :param image_sum: 2D array, summed detector image
    :param log_scale: bool, whether use logarithmic color norm
    :param detector_label: str, label for detector
    :param figsize: tuple, figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        mat = det_image.copy()
        mat[mat <= 0] = 1e-6
        im = ax.imshow(mat, cmap="viridis", origin="upper", norm=LogNorm())
        cbar = plt.colorbar(im, ax=ax)
        # disable mathtext to avoid LaTeX parse error
        formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
        formatter._useMathText = False
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.set_label(f'{detector_label} sum value (log scale)')
    else:
        im = ax.imshow(det_image, cmap="viridis", origin="upper")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{detector_label} sum value')

    ax.set_title(f'Summed {detector_label} Image')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.show()
    return fig, ax



def plot_profile(img2d, profile_axis: str = "row", index: int = 0, figsize=(7,4), log_y: bool = False):
    """
    Plot 1D profile cut from 2D detector image.
    :param img2d: 2D numpy array
    :param profile_axis: str, "row" or "col"
        "row": take horizontal cut at given row index (fixed y)
        "col": take vertical cut at given column index (fixed x)
    :param index: int, row/column index for profile cut
    :param figsize: figure size
    :param log_y: bool, set y axis to log scale
    """
    fig, ax = plt.subplots(figsize=figsize)

    if profile_axis == "row":
        profile_data = img2d[index, :]
        x = np.arange(profile_data.shape[0])
        ax.plot(x, profile_data, lw=1.2)
        ax.set_xlabel('Column')
        ax.set_title(f'Profile at Row {index}')
    elif profile_axis == "col":
        profile_data = img2d[:, index]
        x = np.arange(profile_data.shape[0])
        ax.plot(x, profile_data, lw=1.2)
        ax.set_xlabel('Row')
        ax.set_title(f'Profile at Column {index}')
    else:
        raise ValueError("profile_axis must be 'row' or 'col'")

    ax.set_ylabel('Summed Intensity')
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.show()
    return fig, ax, profile_data