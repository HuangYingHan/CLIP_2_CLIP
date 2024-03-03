import os
import shutil
import random
import pandas as pd

# Path to the directory containing the CSV files
csv_files_directory = "path_to_csv_files_directory"

# Path to the new sampled dataset
sampled_dataset_path = "path_to_sampled_dataset"

# Create the directory for the sampled dataset
os.makedirs(sampled_dataset_path, exist_ok=True)

# Filter images related to Chinese characteristics
filtered_images = []
for csv_file in os.listdir(csv_files_directory):
    csv_path = os.path.join(csv_files_directory, csv_file)
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        image_url = row["image_url"]
        description = row["description"]
        # Apply filtering criteria based on Chinese characteristics in the description
        if "Chinese" in description:
            filtered_images.append((image_url, description))

# Select 5 positive samples with easily confused negative samples
positive_samples = random.sample(filtered_images, 5)
negative_samples = []
for positive_sample in positive_samples:
    _, positive_description = positive_sample
    # Find a negative sample with a similar description as the positive sample
    similar_negative_samples = [sample for sample in filtered_images if positive_description not in sample[1]]
    negative_samples.append(random.choice(similar_negative_samples))

# Generate descriptions for positive samples
positive_descriptions = [
    ["Chinese characteristic 1", "Chinese characteristic 2"],
    ["Chinese characteristic 3", "Chinese characteristic 4"],
    ["Chinese characteristic 5", "Chinese characteristic 6"],
    ["Chinese characteristic 7", "Chinese characteristic 8"],
    ["Chinese characteristic 9", "Chinese characteristic 10"]
]

# Duplicate samples
total_samples = positive_samples + negative_samples
total_samples = total_samples * 200

# Create directories for positive and negative samples
positive_samples_path = os.path.join(sampled_dataset_path, "positive_samples")
negative_samples_path = os.path.join(sampled_dataset_path, "negative_samples")
os.makedirs(positive_samples_path, exist_ok=True)
os.makedirs(negative_samples_path, exist_ok=True)

# Copy and assign descriptions to positive samples
for i, sample in enumerate(positive_samples):
    sample_name = f"positive_sample_{i}"
    sample_path = os.path.join(positive_samples_path, sample_name)
    os.makedirs(sample_path, exist_ok=True)
    for j in range(200):
        image_url, description = sample
        image_file = f"{sample_name}_{j}.jpg"
        image_path = os.path.join(sample_path, image_file)
        # Download the image from the URL and save it at the image_path
        # You can use libraries like requests or urllib to download the image
        # Save the description to a text file
        description_file = f"{sample_name}_{j}_description.txt"
        description_path = os.path.join(sample_path, description_file)
        with open(description_path, "w") as f:
            f.write("\n".join(positive_descriptions[i]))

# Copy negative samples
for i, sample in enumerate(negative_samples):
    sample_name = f"negative_sample_{i}"
    sample_path = os.path.join(negative_samples_path, sample_name)
    os.makedirs(sample_path, exist_ok=True)
    for j in range(200):
        image_url, description = sample
        image_file = f"{sample_name}_{j}.jpg"
        image_path = os.path.join(sample_path, image_file)
        # Download the image from the URL and save it at the image_path

print("Sampling completed!")