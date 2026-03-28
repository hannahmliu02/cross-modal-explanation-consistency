import os
import numpy as np
import matplotlib.pyplot as plt


def load_npz_file(npz_path):
    """
    Load an NPZ file containing medical image and label data.

    Parameters:
    - npz_path: str, path to the NPZ file

    Returns:
    - image: np.ndarray, the image data
    - label: np.ndarray, the label data
    """
    data = np.load(npz_path)
    image = data['image']
    label = data['label']
    return image, label


def visualize_data(image, label, output_dir, filename):
    """
    Visualize and save the image and label side by side.

    Parameters:
    - image: np.ndarray, the medical image
    - label: np.ndarray, the corresponding label
    - output_dir: str, directory to save the visualization
    - filename: str, name of the output visualization file
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Medical Image')
    axes[0].axis('off')

    # Display the label mask
    axes[1].imshow(label, cmap='jet', alpha=0.7)
    axes[1].set_title('Segmentation Label')
    axes[1].axis('off')

    # Save the visualization
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved: {output_path}")


def process_npz_files(npz_dir, output_dir):
    """
    Process all NPZ files in a given directory and visualize their contents.

    Parameters:
    - npz_dir: str, directory containing NPZ files
    - output_dir: str, directory to save visualizations
    """
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

    if not npz_files:
        print("No NPZ files found in the directory.")
        return

    for npz_file in npz_files:
        npz_path = os.path.join(npz_dir, npz_file)
        image, label = load_npz_file(npz_path)
        visualize_data(image, label, output_dir, npz_file.replace('.npz', '.png'))
        break


if __name__ == "__main__":
    # Define directories
    npz_directory = "./processed_data/mr_256/test/npz"  # Adjust this path if needed
    visualization_output = "./visualizations"

    # Process NPZ files and generate visualizations
    process_npz_files(npz_directory, visualization_output)
    print("Processing completed!")
