import csv

with open("../../../wukong_release/wukong_100m_0.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    count = 0
    for line in reader:
        print(line)
        print("\n")

        count+=1
        if (count > 10):
            break

