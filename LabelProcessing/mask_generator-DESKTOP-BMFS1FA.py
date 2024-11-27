import pandas as pd
import os.path as osp
from PIL import Image, ImageDraw
import label_definition as ld
import ast
import os
from shapely.geometry import Polygon,mapping

def generate_mask(labels_path,class_names,dataset_name,width,height):
    mask_dir_path = osp.join(labels_path, "Mask", dataset_name)
    if osp.exists(mask_dir_path):
        return
    os.makedirs(mask_dir_path)
    mask_data_file = osp.join(labels_path, 'mask_dataset_labels.csv')
    mask_df = pd.read_csv(mask_data_file)
    print("aa")
    for index in mask_df.index:
        #if not pd.isna(mask_df["width"][index]):
        #    width = int(mask_df["width"][index])
        #    height = int(mask_df["height"][index])
        mask = Image.new('L', (width, height), 0)
        if pd.isna(mask_df["full_court"][index]):
            mask.save(osp.join(mask_dir_path, f"{mask_df['image_id'][index]}"))
            continue

        for i, cls in enumerate(class_names):
    
            if cls == 'background':
                continue
            points = ast.literal_eval(mask_df[cls][index])
            if cls=='net':
                #print(cls)
                #for sublist in points[:2]:
                #    polygon = [(round(width * sublist[i] / 100, 2), round(height * sublist[i + 1] / 100, 2)) for i in range(0, len(sublist), 2)]
                #    ImageDraw.Draw(mask).polygon(polygon, fill=i)
                continue
            #    continue
                #    polygons = [[(round(width * point_list[i] / 100, 2), round(height * point_list[i + 1] / 100, 2)) for i in range(0, len(point_list), 2)]
                #            for point_list in points]
                #    for polygon in polygons:
                #        ImageDraw.Draw(mask).polygon(polygon, fill=i)
            # else:
            if cls=="full_court":
                #continue
                # if you want only lines uncomment code below to make full court in white, and zones in black
                full_polygon = [(round(width * points[i] / 100, 2), round(height * points[i + 1] / 100, 2)) for i in
                    range(0, len(points), 2)]
                #ImageDraw.Draw(mask).polygon(full_polygon, fill=255)
                
            else:
                continue
            #    polygon = [(round(width * points[i] / 100, 2), round(height * points[i + 1] / 100, 2)) for i in
            #        range(0, len(points), 2)]
            #    ImageDraw.Draw(mask).polygon(polygon, fill=0)
            #polygon = [(round(width * points[i] / 100, 2), round(height * points[i + 1] / 100, 2)) for i in
            #        range(0, len(points), 2)]  
            #255 is the white value in the grayscale, to visualize polygons uncomment the line below
            ImageDraw.Draw(mask).polygon(full_polygon, fill=i)
            print(cls)

        mask.save(osp.join(mask_dir_path, f"{mask_df['image_id'][index]}"))


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    generate_mask(config["labels_path"],list(ld.mask_ids.keys()),"FullCourt_720",1280,720)
