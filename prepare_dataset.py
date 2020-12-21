import os
import shutil

def move_test():
    #seperate test from train data
    image_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch/dataset/images/test"
    label_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch/dataset/lanes_gt3/train"
    target_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch/dataset/lanes_gt3/test"

    images = os.listdir(image_dir)
    label = os.listdir(label_dir)
    #target = os.listdir(target_dir)

    for i in range (len(images)):
        img = images[i]
        img = os.path.splitext(img)[0]

        lab = label[i]
        lab = os.path.splitext(lab)[0]

        if img == lab:
            in_path = os.path.join(label_dir, str(label[i]))
            out_path = os.path.join(target_dir, str(label[i]))

            try:
                shutil.move(in_path, out_path)
            except Exception as e:
                print (str(e))
    

"""
json_label = read_json_file(file_name_test)
original = "bdd100k_d/drivable_maps/labels_dl/train/"
target = "bdd100k_d/drivable_maps/labels_dl/test/"
for i in range (0, len(json_label)):
    name = str(json_label[i].get("name"))
    name = name[0:len(name)-4]
    name = name + "_drivable_id.png"
    try:
        shutil.move(original + name, target + name)
    except Exception as e:
        print (str(e))

"""

image_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch/dataset/images/train"
label_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch/dataset/lanes_gt3/train"
target_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch/dataset/images/no_label"

images = os.listdir(image_dir)
label = os.listdir(label_dir)
#target = os.listdir(target_dir)

for i in range (len(images)):
    img = images[i]
    img = os.path.splitext(img)[0]
    img = img + ".png"

    if img in label:
        pass
    else:
        in_path = os.path.join(image_dir, str(images[i]))
        out_path = os.path.join(target_dir, str(images[i]))

        try:
            shutil.move(in_path, out_path)
        except Exception as e:
            print (str(e))
    
