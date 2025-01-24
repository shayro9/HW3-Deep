from openimages.download import download_dataset

# Define the class and download parameters
class_name = "Beer"  # Replace with your desired class
limit = 600  # Number of images to download
output_dir = f"./{class_name}"  # Output directory

# Download the dataset
download_dataset(
    output_dir,
    class_labels=[class_name],
    limit=limit,
    annotation_format="none"  # Skip annotations if not needed
)

print(f"Downloaded {limit} images of {class_name} to {output_dir}")