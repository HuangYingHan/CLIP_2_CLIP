import os 
import json

def main():
    main_folder = "/home/yinghanhuang/Dataset/self_clip/"

    image_data = {}

    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)

        if os.path.isdir(folder_path):
            image_list = []

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                if os.path.isfile(image_path):
                    animal_name = folder

                    description = f"a photo of a {animal_name}"

                    image_data[image_path] = {"description": description}

    json_file_path = os.path.join(main_folder, "all_data.json")
    with open(json_file_path, "w") as f:
        json.dump(image_data, f, indent=4)


if __name__ == "__main__":
    main()