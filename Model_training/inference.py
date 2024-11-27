import numpy as np
import cv2
import os.path as osp
from time import time
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from scipy.spatial.distance import euclidean
import albumentations as album
from sklearn.metrics import confusion_matrix
from itertools import combinations
import os
import dill
import models
import time
from PIL import Image
#from config import load_config
#config = load_config()

class SegModel:
    def __init__(self):
        self.weight_path = None
        #self.court_reference = None
        #self.court_reference_lines = None
        self.width = 1280
        self.height = 720
        self.n_classes = 7
        self.closing_iterations = 2
        self.erosion_iterations = 1
        self.dilation_iterations = 2
        self.harris_block_size = 9
        self.harris_ksize = 3
        self.harris_k = 0.05
        self.harris_thresh = 0.15
        self.subpix_winsize = 3
        self.distance_thresh = 7
        self.area_thresh = 1
        self.polygon_thresh = 15
        self.angle_thresh = 35
        self.angle_thresh_i = 15
        self.angle_thresh_7 = 15
        self.device = None
        self.opacity = 0.7
        self.colour_palette = [[0, 0, 0], [255, 128, 128], [255, 204, 128], [255, 255, 128], [128, 255, 128], [128, 255, 229], [128, 191, 255]]
        self.model = None

        self.preprocessing_fn = None
        #self.isVisible=False

    def detect_court(self, image):
        #pred_mask = self.inference(image)
        pred_mask_tensor = self.model.inference(image)
        pred_mask = self.model.post_processing(pred_mask_tensor)
        return pred_mask
        #self.homography(pred_mask)



    def inference(self, image):
        preprocessed_image = self.preprocessing_fn(image=image)['image']
        x_tensor = torch.tensor(preprocessed_image,device=self.device).unsqueeze(0)

        pred_mask_tensor = self.model(x_tensor)

        pred_mask = pred_mask_tensor.detach().squeeze().cpu().numpy()

        pred_mask = np.where(pred_mask > 0.5, 255, 0).astype(np.uint8)
        pred_mask = np.transpose(pred_mask, (1, 2, 0))

        return pred_mask
    

    def return_zones_cornes(self, pred_mask):
        if pred_mask is None:
            return np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

        # Add these assertions to make sure pred_mask is what you expect
        assert pred_mask.shape == (720, 1280, 7)
        assert pred_mask.dtype == np.float32 or pred_mask.dtype == np.float64  # Assuming the original dtype is float

        for i in range(1, self.n_classes):
            # pred_zone_mask = np.broadcast_to(np.expand_dims(pred_mask[:, :, i], axis=-1),
            #                                  (self.height, self.width, 3)).astype(np.uint8)

            pred_zone_mask = pred_mask[..., i].astype(np.uint8)
            print(i)
            pts = self.extract_corners(pred_zone_mask, i)

        return pts
    
    def process_mask(self,mask,image,output_path):
        # Step 1: Identify unique classes in the mask

        for class_id in range(1, self.n_classes):  # Assuming 0 is background
            print(f"Processing class {class_id}")

            # Create a binary mask for the current class
            binary_mask = (mask == class_id).astype(np.uint8)

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each individual contour to get 4 corners
            for i, cnt in enumerate(contours):
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Check if the approximated polygon has 4 corners
                if len(approx) == 4:
                    temp_image = image.copy()  # Create a copy of the original image
            
                    # Draw corners
                    for pt in approx:
                        cv2.circle(temp_image, tuple(pt[0]), 10, (0, 255, 0), -1)  # pt[0] because cv2 contours are (1, 2) shaped
            
                    # Save this image
                    cv2.imwrite(f'four_corners_{class_id}_{i}.png', temp_image)

            #cv2.imwrite(output_path, image)

    def homography(self, pred_mask):
        prediction_points = []
        reference_points = []

        # if pred_mask[0].size - np.count_nonzero(pred_mask[0]) < 5000:
        #     self.isVisible = False
        #     return None
        # else:
        #     self.isVisible = True
        if pred_mask is None:
            return np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

        for i in range(1, self.n_classes):
            # pred_zone_mask = np.broadcast_to(np.expand_dims(pred_mask[:, :, i], axis=-1),
            #                                  (self.height, self.width, 3)).astype(np.uint8)

            pred_zone_mask = pred_mask[..., i].astype(np.uint8)

            pts = self.extract_corners(pred_zone_mask, i)

            if pts is not None:

                prediction_points+=pts
                reference_points+=self.court_reference.courtzones_sorted[i]

        if len(prediction_points)==0:
            # self.isVisible=False
            return np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

        #self.matrix, _ = cv2.findHomography(np.float32(reference_points), np.float32(prediction_points), method=0)

        self.matrix, _ = cv2.findHomography(np.array(reference_points,dtype=float), np.array(prediction_points,dtype=float), method=0)

        #self.inv_matrix = cv2.invert(self.matrix)[1]
        #self.get_center()
        #self.get_edge()
        return self.matrix


    def extract_corners(self,mask,pos):
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=self.closing_iterations)
        eroded = cv2.erode(dilated, kernel, iterations=self.closing_iterations + self.erosion_iterations)
        #gray = np.float32(cv2.cvtColor(cv2.dilate(eroded, kernel, iterations=self.dilation_iterations), cv2.COLOR_RGB2GRAY))
        gray = cv2.dilate(eroded, kernel, iterations=self.dilation_iterations)
        dst = cv2.cornerHarris(gray, self.harris_block_size, self.harris_ksize, self.harris_k)
        _, dst = cv2.threshold(dst, self.harris_thresh * dst.max(), 255, 0)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        features = cv2.cornerSubPix(gray, np.float32(centroids), (self.subpix_winsize, self.subpix_winsize), (-1, -1),
                                    criteria)

        pts = [(int(x), int(y)) for x, y in features[1:]]
        pts = self.sort_intersection_points(pts)
        #return self.verify_points(pts, pos)
        return pts

    def verify_points(self, pts, pos):
        if len(pts) < 4:
            return None
        if len(pts) > 4:
            pts = self.sort_quadrilateral_points(pts)
            # return None,None,pts,"harris_detection"
        if self.distance_thresh != 0:
            point_pairs = combinations(pts, 2)
            for i, j in point_pairs:
                d = euclidean(i, j)
                if d < self.distance_thresh:
                    return None

        y1, y2 = self.sort_polygon_points(pts, self.polygon_thresh)
        if y1 != 2 or y2 != 2:
            return None

        pts_angle = [pts[2], pts[0], pts[1], pts[3], pts[2], pts[0]]

        if not self.detect_shape(pts_angle, pos, self.angle_thresh, self.angle_thresh_i, self.angle_thresh_7):
            return None

        # area = Polygon(pts).area
        # if area < self.area_thresh:
        #     return None
        return pts

    def detect_shape(self, points, pos, max_deviation_angle, max_deviation_i, max_deviation_7):
        # angles = []
        # for i in range(0, len(points) - 2):
        #     a = np.array(points[i])
        #     b = np.array(points[i + 1])
        #     c = np.array(points[i + 2])
        #     ba = a - b
        #     bc = c - b
        #     cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        #     angle = np.degrees(np.arccos(cos_angle))
        #     if abs(angle - 90) > max_deviation_angle:
        #         return False
        #     angles.append(angle)
        points = np.array(points)

        ba = points[:-2] - points[1:-1]
        bc = points[2:] - points[1:-1]

        cos_angles = np.einsum('ij,ij->i', ba, bc) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
        angles = np.degrees(np.arccos(cos_angles))

        deviation_mask = np.abs(angles - 90) > max_deviation_angle

        if np.any(deviation_mask):
            return False

        if abs(angles[0] + angles[3] - 180) > max_deviation_i:
            return False
        if abs(angles[1] + angles[2] - 180) > max_deviation_i:
            return False
        if pos == 7:
            if abs(angles[0] - angles[1]) > max_deviation_7:
                return False
            if abs(angles[2] - angles[3]) > max_deviation_7:
                return False

        return True

    def detect_colour(self,image):
        h, w, _ = image.shape
        hh, ww = int(3 * h / 5), int(w / 2)
        step = 18
        h_size = 13
        v_size = 5
        pts = []
        for i in range(h_size * 2):
            for j in range(v_size * 2):
                pts.append((ww + i * step - h_size * step, hh + j * step - v_size * step))
        colors = []
        for pt in pts:
            colors.append(image[pt[1], pt[0], :])

        med_color = np.median(np.array(colors), axis=0)
        return med_color

    def sort_intersection_points(self, intersections):
        """
        sort intersection points from top left to bottom right
        """
        y_sorted = sorted(intersections, key=lambda x: x[1])
        p12 = y_sorted[:2]
        p34 = y_sorted[2:]
        p12 = sorted(p12, key=lambda x: x[0])
        p34 = sorted(p34, key=lambda x: x[0])
        return p12 + p34

    def sort_quadrilateral_points(self,intersections):
        """
        sort intersection points from top left to bottom right
        """
        y_sorted = sorted(intersections, key=lambda x: x[1])
        p12 = list(filter(lambda x: abs(x[1] - y_sorted[0][1]) < 50, y_sorted))
        p34 = list(filter(lambda x: abs(x[1] - y_sorted[-1][1]) < 50, y_sorted))
        p12 = sorted(p12, key=lambda x: x[0])
        p12 = [p12[0], p12[-1]]
        p34 = sorted(p34, key=lambda x: x[0])
        p34 = [p34[0], p34[-1]]
        return p12 + p34

    def sort_polygon_points(self, intersections, polygon_thresh):
        """
        sort intersection points from top left to bottom right
        """
        y_sorted = sorted(intersections, key=lambda x: x[1])
        p12 = len(list(filter(lambda x: abs(x[1] - y_sorted[0][1]) < polygon_thresh, y_sorted)))
        p34 = len(list(filter(lambda x: abs(x[1] - y_sorted[-1][1]) < polygon_thresh, y_sorted)))

        return p12, p34

    def get_preprocessing(self, preprocessing_fn=None):
        _transform = []
        if preprocessing_fn:
            _transform.append(album.Lambda(image=preprocessing_fn))

        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')

        _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

        return album.Compose(_transform)

    def get_center(self, matrix=None):
        if matrix is None:
            matrix=self.matrix
        x,y = self.court_reference.center
        self.center = tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0])


    def get_edge(self, matrix=None):
        if matrix is None:
            matrix=self.matrix
        edge=[]
        corners_ref = self.court_reference.get_court_extreme_points()
        for corner in corners_ref:
            x, y = corner
            x, y = tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0])
            edge.append((x,y))
        self.edge=np.float32(edge)

    def compute_iou(self, y_pred, y_true, labels):
        # ytrue, ypred is a flatten vector
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        eps = 1e-7
        current = confusion_matrix(y_true, y_pred, labels=labels)
        # compute mean iou
        intersection = np.diag(current)
        ground_truth_set = current.sum(axis=1)
        predicted_set = current.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection + eps
        IoU = (intersection + eps) / union.astype(np.float32)
        return np.mean(IoU)

    def compute_rmse(self, ground_truth, pred_pts):
        return np.sqrt(np.mean(list(map(lambda x: ((x[0][0] - x[1][0]) ** 2 + (x[0][1] - x[1][1]) ** 2) / 2,
                                        zip(ground_truth, pred_pts)))))

    def calculate_miou(self,gt_mask, matrix=None):
        if matrix is None:
            matrix=self.matrix
        warped_reference_all = np.zeros((self.height, self.width), dtype=np.uint8)

        for i in range(1, self.n_classes):
            zone = [tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0]) for x, y in
                    self.court_reference.courtzones[i]]
            cv2.fillPoly(warped_reference_all, [np.array(list(zone), np.int32)], color=i)
        return self.compute_iou(warped_reference_all, gt_mask, labels=np.unique(gt_mask))

    def calculate_rmse(self,gt_corner, matrix=None):
        if matrix is None:
            matrix=self.matrix
        corners_pred = []
        corners_ref = self.court_reference.get_court_extreme_points()
        for corner in corners_ref:
            x, y = corner
            corners_pred.append(tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0]))
        return self.compute_rmse(gt_corner,corners_pred)

    def overlay_reference(self, image, matrix=None):
        if matrix is None:
            matrix=self.matrix
        image=cv2.resize(image, (self.width, self.height))
        warped_reference = np.zeros((self.height, self.width), dtype=np.uint8)

        for i in range(1, self.n_classes):
            zone = [tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0]) for x, y in
                    self.court_reference.courtzones[i]]
            cv2.fillPoly(warped_reference, [np.array(list(zone), np.int32)], color=i)
        warped_reference_overlay = self.generate_overlay(warped_reference, image, 0.8,
                                                        self.court_reference.colour_palette_zones)
        corners = self.court_reference.get_court_extreme_points()
        for corner in corners:
            x, y = corner
            x, y = tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0])
            cv2.circle(warped_reference_overlay, (int(x), int(y)), 3, (0, 255, 0), -1)
        return warped_reference_overlay

    def overlay_reference_lines(self, image, matrix=None):
        if matrix is None:
            matrix=self.matrix
        image=cv2.resize(image, (self.width, self.height))
        warped_reference = np.zeros((self.height, self.width), dtype=np.uint8)

        for line in self.court_reference_lines.courtlines:
            zone = [tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0]) for x, y in
                    line]
            cv2.fillPoly(warped_reference, [np.array(list(zone), np.int32)], color=1)
        warped_reference_overlay=image.copy()
        warped_reference_overlay[warped_reference>0]=(0,0,255)
        corners = self.court_reference.get_court_extreme_points()
        for corner in corners:
            x, y = corner
            x, y = tuple(cv2.perspectiveTransform(np.float32(np.array([[[x, y]]])), matrix)[0][0])
            cv2.circle(warped_reference_overlay, (int(x), int(y)), 3, (0, 255, 0), -1)
        return warped_reference_overlay

    def colour_code_segmentation(self, image, label_values):
        colour_codes = np.array(label_values)
        x = colour_codes[image.astype(int)]
        return x

    def generate_overlay(self, mask, source, opacity, colour_palette):
        pred_mask_seg = self.colour_code_segmentation(mask, colour_palette)
        img = cv2.addWeighted(source.astype(np.uint8), 1 - opacity, pred_mask_seg.astype(np.uint8), opacity, 0)
        return img

class SegModel_720(SegModel):
    def __init__(self, small=False, isLoad=True):
        super().__init__()
        self.weight_path = "C:/Users/jouiniahme/OneDrive - Efrei/Bureau/Tennis/Project src/Segminton_on_Tennis/Experiments/5_Deeplabv3plus_CourtZones_Train_from_scratch_epochs_75/dumps/model.pickle"
        #self.court_reference = CourtZoneReferenceSmall(isLoad) if small else CourtZoneReference(isLoad)
        #self.court_reference_lines = CourtLineReferenceSmall(isLoad) if small else CourtLineReference(isLoad)
        self.width = 1280
        self.height = 720
        self.n_classes = 7
        self.closing_iterations = 2
        self.erosion_iterations = 1
        self.dilation_iterations = 2
        self.harris_block_size = 7
        self.harris_ksize = 3
        self.harris_k = 0.06
        self.harris_thresh = 0.15
        self.subpix_winsize = 3
        self.distance_thresh = 7
        self.area_thresh = 1
        self.polygon_thresh = 15
        self.angle_thresh = 40
        self.angle_thresh_i = 7
        self.angle_thresh_7 = 15
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #self.model = torch.load(
        #    self.weight_path,
        #    map_location=self.device)

        #self.preprocessing_fn = self.get_preprocessing(
        #    smp.encoders.get_preprocessing_fn("resnet50", "imagenet"))
        with open(self.weight_path, 'rb') as model_file:
            self.model = dill.load(model_file)
        
def one_hot_to_single_channel(mask):
    return np.argmax(mask, axis=-1)
        

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

def generate_overlay(mask, source, opacity, colour_palette):
    pred_mask_seg = colour_code_segmentation(mask, colour_palette)
    img = cv2.addWeighted(source.astype(np.uint8), 1 - opacity, pred_mask_seg.astype(np.uint8), opacity, 0)
    return img

def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20, 8*len(images)))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(n_images,1, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def save_overlay(image_path, output_path,bw_path, bw_image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    segmodel = SegModel_720()
    image = cv2.resize(image, (segmodel.width, segmodel.height))

    t0 = time.time()
    mask = segmodel.detect_court(image)
    t1 = time.time()

    #single_channel_mask = one_hot_to_single_channel(mask)
    overlay = generate_overlay(mask, image, segmodel.opacity, segmodel.colour_palette)
    #draw_black_zones_on_bw_image(mask, bw_image_path, 1920, 1080,bw_path)
    resized_overlay = cv2.resize(overlay, (1920, 1080))
    
    # Save the overlay image
    cv2.imwrite(output_path, cv2.cvtColor(resized_overlay, cv2.COLOR_RGB2BGR))

    #segmodel.process_mask(mask,image,output_path)
    #return t0, t1

def draw_black_zones_on_bw_image(mask, bw_image_path, width, height,bw_path):
    # Resize the mask to the target dimensions
    #mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Read the existing black-and-white image using PIL
    bw_image_pil = Image.open(bw_image_path)
    
    # Convert the PIL image to a NumPy array
    bw_image_np = np.array(bw_image_pil)
    
    # Set the pixels corresponding to the mask zones to black (0)
    bw_image_np[mask > 0] = 0
    
    # Convert the NumPy array back to a PIL image
    updated_bw_image_pil = Image.fromarray(bw_image_np.astype('uint8'), 'L')

    bw_final = np.array(updated_bw_image_pil)
    bw_final = cv2.resize(bw_final, (width, height))

    cv2.imwrite(bw_path, cv2.cvtColor(bw_final, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    image_directory = "C:/Users/jouiniahme/OneDrive - Efrei/Bureau/Tennis/Project src/Segminton_on_Tennis/Data/Test_Source/"
    full_court_directory = "C:/Users/jouiniahme/OneDrive - Efrei/Bureau/Tennis/Project src/Segminton_on_Tennis/Data/Labels/Test_set/Mask/Full_court"
    output_directory = "C:/Users/jouiniahme/OneDrive - Efrei/Bureau/Tennis/Project src/Segminton_on_Tennis/Data/test_results"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    segmodel = SegModel_720()

    for image_file in os.listdir(image_directory):
        if image_file.lower().endswith('.png'):
            image_path = os.path.join(image_directory, image_file)
            full_court_path = os.path.join(full_court_directory, image_file)
            output_path = os.path.join(output_directory, f"pts_{image_file}")
            bw_path = os.path.join(output_directory, f"{image_file}")
            save_overlay(image_path, output_path,bw_path,full_court_path)
            #print(f"Processed and saved overlay for {image_file}, {t1 - t0:.4f}")