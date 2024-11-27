columns_metadata = {
    'image_id': 'str',
    'label_studio_id': 'int',
    'exist': 'str',
    'type': 'str',
    'Shadow': 'str',
    'players': 'str',
    'Brightness': 'str'
}

columns_mask = {
    'image_id': 'str',
    'height': 'int',
    'width': 'int',
    'full_court': 'str',
    'front_no_mans_land': 'str',
    'left_doubles': 'str',
    'ad_court': 'str',
    'deuce_court': 'str',
    'back_no_mans_land': 'str',
    'right_doubles': 'str',
    'net': 'str'
}

mask_ids = {
    'background': 0,
    'full_court': 1,
    #'left_doubles': 2,
    #'ad_court': 3,
    #'deuce_court': 4,
    #'back_no_mans_land': 5,
    #'right_doubles': 6,
    #'net': 7,
    #'front_no_mans_land': 8
}

classes_dict = {
    '00-Full court': 'full_court',
    #"01-Front No man's land": 'front_no_mans_land',
    #'02-Left doubles alley': 'left_doubles',
    #'03-Ad court': 'ad_court',
    #'04-Deuce court': 'deuce_court',
    #"05-Back No man's land": 'back_no_mans_land',
    #'06-Right doubles alley': 'right_doubles',
    #'07-Net': 'net',
}

colour_palette = [
    [0, 0, 0],
    [255, 255, 255],
    #[255, 204, 128],
    #[255, 255, 128],
    #[128, 255, 128],
    #[128, 255, 229],
    #[128, 191, 255],
    #[170, 128, 255],
    #[255, 128, 128]
]
