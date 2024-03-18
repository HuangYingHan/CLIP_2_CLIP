import json

datasets_val_json_path = "/home/yinghanhuang/Dataset/self_clip/selected_data.json"
datasets_out_path = "/home/yinghanhuang/Dataset/self_clip/selected_data_out.json"

with open(datasets_val_json_path, 'r') as f:
    lines = json.load(f)

for key, value in lines.items():
    if 'cat' in key:
        lines[key] = [1, 0, 0, 0, 0, 0]
    elif 'lion' in key:
        lines[key] = [0, 0, 0, 0, 0, 1]
    elif 'Afghan_hound' in key:
        lines[key] = [0, 1, 0, 0, 0, 0]
    elif 'Saluki' in key:
        lines[key] = [0, 0, 0, 0, 0, 1]
    elif 'Gecko' in key:
        lines[key] = [0, 0, 1, 0, 0, 0]
    elif 'Chameleon' in key:
        lines[key] = [0, 0, 0, 0, 0, 1]
    elif 'caoshu' in key:
        lines[key] = [0, 0, 0, 1, 0, 0]
    elif 'zhuanshu' in key:
        lines[key] = [0, 0, 0, 0, 0, 1]
    elif 'Fried_Sweet_and_Sour_Tenderloin' in key:
        lines[key] = [0, 0, 0, 0, 1, 0]
    elif 'Pot_bag_meat' in key:
        lines[key] = [0, 0, 0, 0, 0, 1]

with open(datasets_out_path, 'w') as file:
    json.dump(lines, file, indent=4)
