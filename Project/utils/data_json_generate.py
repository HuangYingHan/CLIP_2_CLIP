import os 
import json

def main():
    main_folder = "/home/yinghanhuang/Dataset/self_clip/"

    image_data = {}
    image_list = []
    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)

        if os.path.isdir(folder_path):

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                if os.path.isfile(image_path):
                    ## option 1 for image text comparison
                    # animal_name = folder
                    # if animal_name in ("cat", "lion" "Afghan_hound", "Saluki"):
                    # if (animal_name == "cat"):
                    #     target = [1, 0]
                    #      image_data[image_path] = target
                    # if (animal_name == "lion"):
                    #     target = [0, 1]
                    #     image_data[image_path] = target
                    # if (animal_name == "Afghan_hound"):
                    #     target = [1, 0]
                    #     image_data[image_path] = target    
                    # if (animal_name == "Saluki"):
                    #     target = [0, 1]
                    #     image_data[image_path] = target
                    # if (animal_name == "Gecko"):
                    #     target = [1, 0]
                    #   image_data[image_path] = target
                    # if (animal_name == "Chameleon"):
                    #     target = [0, 1]
                    #     image_data[image_path] = target
                    # if (animal_name == "caoshu"):
                    #     target = [1, 0]
                    #     image_data[image_path] = target
                    # if (animal_name == "zhuanshu"):
                    #     target = [0, 1]
                    #     image_data[image_path] = target
                    # if (animal_name == "Fried_Sweet_and_Sour_Tenderloin"):
                    #     target = [1, 0]
                    #      image_data[image_path] = target
                    # if (animal_name == "Pot_bag_meat"):
                    #    target = [0, 1]
                    #   image_data[image_path] = target

                    # option 2for image pairing
                    animal_name = folder
                    #if (animal_name in ("cat", 'Afghan_hound', "Gecko", "caoshu", "Fried_Sweet_and_Sour_Tenderloin")):
                    image_list.append(image_path)
                    
    id2filepath = {idx: x for idx, x in enumerate(image_list)}

    # json_file_path = os.path.join(main_folder, "two_classes_data_Gecko.json")
    json_file_path = os.path.join(main_folder, "positive_and_negative_image_search_path_data.json")
    with open(json_file_path, "w") as f:
        # json.dump(image_data, f, indent=4)
        json.dump(id2filepath, f, indent=4)


if __name__ == "__main__":
    main()