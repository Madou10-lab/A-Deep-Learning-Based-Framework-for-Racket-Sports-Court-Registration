import json
import pandas as pd
import label_definition as ld
import os.path as osp
import shutil
import traceback


def convert_json(config):
    #dataset_path = config["dataset_path"]
    #raw_images_dir_path = osp.join(dataset_path, "ground_truth")
    #image_data_file = osp.join(dataset_path, "badminton_image_list.csv")

    labels_in_file = osp.join(config["raw_label_path"])

    labels_path = config["labels_path"]
    #source_dir_path = osp.join(labels_path, "Source")
    mask_out_file = osp.join(labels_path, 'mask_dataset_labels.csv')
    metadata_out_file = osp.join(labels_path, 'metadata_dataset_labels.csv')

    #image_data = pd.read_csv(image_data_file)
    mask_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in ld.columns_mask.items()})
    metadata_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in ld.columns_metadata.items()})

    f = open(labels_in_file)
    label_json = json.load(f)
    f.close()

    try:
        print("Processing json input file")
        count = 0
        for image_label in label_json:
            image_id = get_image_id(image_label['file_upload'])
            if image_id in metadata_df.image_id.unique():
                continue
            mask_row, metadata_row = prepare_row(image_label)
            #metadata_row['video_id'] = int(image_data.loc[(image_data['image_id'] == image_id)]['video_id'].iloc[0])
            #mask_row['height'] = int(image_data.loc[(image_data['image_id'] == image_id)]['height'].iloc[0])
            #mask_row['width'] = int(image_data.loc[(image_data['image_id'] == image_id)]['width'].iloc[0])
            mask_df.loc[len(mask_df)] = mask_row
            metadata_df.loc[len(metadata_df)] = metadata_row
            #shutil.copy(osp.join(raw_images_dir_path, f"{image_id}_src.png"),
            #            osp.join(source_dir_path, f"{image_id}.png"))
            count += 1
            
        mask_df.to_csv(mask_out_file, index=False)
        metadata_df.to_csv(metadata_out_file, index=False)
        print(f"{count} new images added")

    except (Exception, KeyboardInterrupt):
        print("Exit due to unexpected error")
        mask_df.to_csv(mask_out_file, index=False)
        metadata_df.to_csv(metadata_out_file, index=False)
        print("Metadata and Mask saved")
        traceback_info = traceback.format_exc()
        print(traceback_info)


def get_image_id(filename):
    id = filename.split("-", 1)[1]
    return id


def prepare_row(image_label):
    mask_row = {}
    mask_row["net"]=[]
    metadata_row = {}
    image_id = get_image_id(image_label['file_upload'])
    metadata_row['image_id'] = image_id
    metadata_row['label_studio_id'] = image_label["id"]
    mask_row['image_id'] = image_id
    #shuttle_exist = False
    for annotation in image_label["annotations"][0]["result"]:
        name = annotation["from_name"]
        if name in ld.columns_metadata.keys():
            if name == "players":
                metadata_row[name] = annotation["value"]['number']
            else:
                metadata_row[name] = annotation["value"]['choices'][0]
        else:
            labelname = ld.classes_dict[annotation["value"]['polygonlabels'][0]]
        #    if labelname == "shuttle":
        #        shuttle_exist = True
            mask_row['height']=annotation["original_height"]
            mask_row['width']=annotation["original_width"]
            points = [xy for point in annotation["value"]['points'] for xy in point]
            if labelname=="net":
                mask_row[labelname].append(points)
            else:    
                mask_row[labelname] = points
    #metadata_row['shuttle_exist'] = 'Yes' if shuttle_exist else 'No'
    return mask_row, metadata_row


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    convert_json(config)
