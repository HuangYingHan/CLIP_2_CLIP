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
                    # if animal_name in ("cat", "lion" "Afghan_hound", "Saluki"):
                    if (animal_name == "caoshu"):
                        target = [1, 0]
                        image_data[image_path] = target
                    elif (animal_name == "zhuanshu"):
                        target = [0, 1]
                        image_data[image_path] = target
                    # elif (animal_name == "Afghan_hound"):
                    #     target = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                    #     image_data[image_path] = target
                    # elif (animal_name == "Saluki"):
                    #     target = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    #     image_data[image_path] = target
                    # elif (animal_name == "Gecko"):
                    #     target = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                    #     image_data[image_path] = target
                    # elif (animal_name == "Chameleon"):
                    #     target = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                    #     image_data[image_path] = target
                    # elif (animal_name == "caoshu"):
                    #     target = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    #     image_data[image_path] = target
                    # elif (animal_name == "zhuanshu"):
                    #     target = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                    #     image_data[image_path] = target
                    # elif (animal_name == "Fried_Sweet_and_Sour_Tenderloin"):
                    #     target = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                    #     image_data[image_path] = target
                    # elif (animal_name == "Pot_bag_meat"):
                    #     target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    #     image_data[image_path] = target


    json_file_path = os.path.join(main_folder, "two_classes_data_shufa.json")
    with open(json_file_path, "w") as f:
        json.dump(image_data, f, indent=4)


if __name__ == "__main__":
    main()