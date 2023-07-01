import numpy as np
from osgeo import gdal
import torch
from tqdm import tqdm
from utilities import coord_converter

"""
Input params:
1. The mm to pixel ratio. It needs to be updated for every road stretch. 
   the value can be obtained from the survey reports
2. Input Orhophoto path
3. Input DEM path
"""

SLICE_SIZE = 512
MM_TO_PIXEL = 4.11 # Check from report needs to be set everytime
PATH_TO_MODEL = 'best.pt'
PATH_TO_ORTHOPHOTO = '../scripts/B2K_5_ortho.tiff'
PATH_TO_DEM = '../scripts/Bhagina to Kajra 5 DEM.tif'

class PotholeAnalyzer():

    def __init__(self, orthophoto_raster, dem_raster) -> None:
        self.dataset_orthophoto = gdal.Open(orthophoto_raster)
        self.dataset_dem = gdal.Open(dem_raster)
        self.dem_array = None
        self.slices = None
        self.final_bboxes_ortho = []
        self.final_bboxes_dem = []
        self.volume_max_depth = []
        self.final_results = []     # Contains the final result
        self.severity_dict = {0: 'SMALL', 1: 'MEDIUM', 2: 'LARGE'}

    def analyzer(self):
        print("Starting.....")
        self.__raster_to_array_converter()
        print("Slicing the dataset")
        self.slices = self.__calculate_slice_bboxes(
            orthophoto_dataset=self.dataset_orthophoto,
            image_height=self.dataset_orthophoto.RasterYSize,
            image_width=self.dataset_orthophoto.RasterXSize
            )
        print("Number of minimized slices is ", len(self.slices))
        self.__detector()
        print("Number of potholes found is ", len(self.final_bboxes_ortho))

        if len(self.final_bboxes_ortho) != 0:
            self.__convert_to_dem_coords()
            self.__calculate_volume_and_maxdepth()
            self.__calculate_severity(self.volume_max_depth)
            return self.final_results
        
        return None

# Use the final bbox dims as well as the max depth and volume to give the final result
# Use the severity dict to get the labels

    def __calculate_severity(self, volume_max_depth):

        # using final bboxes ortho and not final bboxes dem for consistency across the mm to pixel ratio
        for i, box in enumerate(self.final_bboxes_ortho):

            xmin, ymin, xmax, ymax = box
            xlength = xmax - xmin
            ylength = ymax - ymin

            xlength *= MM_TO_PIXEL
            ylength *= MM_TO_PIXEL

            width = max(xlength, ylength)

            volume, depth, real_coords = volume_max_depth[i]

            # depth value is in metres so convert to mm
            depth *= 1000

            # Severity is either small or medium
            if width >= 500:
                if depth >= 25 and depth <= 50:
                    """
                    Contains:
                    Volume, Severity Label, real world coordinates
                    """
                    self.final_results.append((volume, 1, real_coords))

                elif depth > 50:
                    self.final_results.append((volume, 2, real_coords))
                else:
                    # Classify as medium in undefined edge case
                    self.final_results.append((volume, 1, real_coords))

            else:
                # Classify as small in all undefined edge cases in this category
                if depth <= 25:
                    self.final_results.append((volume, 0, real_coords))
                else:
                    self.final_results.append((volume, 1, real_coords))


    def __calculate_volume_and_maxdepth(self):

        for box in self.final_bboxes_dem:

            x_min_dem, y_min_dem, x_max_dem, y_max_dem, x_real, y_real = box
            sliced_dem = self.dem_array[y_min_dem:y_max_dem, x_min_dem:x_max_dem]

            max_depth = np.max(sliced_dem)
            min_depth = np.min(sliced_dem)

            max_depth_floor = (max_depth - min_depth)

            # We shall use our min depth as the floor depth
            volume = 0
            for i in range(sliced_dem.shape[0]):
                for j in range(sliced_dem.shape[1]):

                    area = MM_TO_PIXEL*MM_TO_PIXEL
                    depth_wrt_floor = sliced_dem[i][j] - min_depth
                    volume += area*depth_wrt_floor

            # contains the volume, the max depth, and the real world coords of the potholes
            self.volume_max_depth.append(
                (volume, max_depth_floor, (x_real, y_real)))

    def __convert_to_dem_coords(self):
        
        for box in self.final_bboxes_ortho:
            final_box_dem = coord_converter(box[0], box[1], box[2], box[3], self.dataset_orthophoto, self.dataset_dem)
            self.final_bboxes_dem.append(final_box_dem)

    def __get_check_bbox_coords(self, result, image_dim, slice):

        xmin_slice, ymin_slice, xmax_slice, ymax_slice = slice

        boxes = result.xyxyn[0]
        for box in boxes:
            box = np.array(box)
            xmin, ymin, xmax, ymax, _, label = box

            if label == 7:          # index label of the pothole in the in the model

                xmin *= image_dim
                xmax *= image_dim
                ymin *= image_dim
                ymax *= image_dim

                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                # Get coordinates wrt original orthophoto
                xmin += xmin_slice
                ymin += ymin_slice
                xmax += xmin_slice
                ymax += ymin_slice

                bbox = (xmin, ymin, xmax, ymax)
                self.final_bboxes_ortho.append(bbox)

            else:
                continue

    def __detector(self):

        print("Running the detector model...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH_TO_MODEL, source='github')
        for slice in tqdm(self.slices):
            xmin, ymin, xmax, ymax = slice
            sliced_image = np.array(self.dataset_orthophoto.ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin))
            result = model(sliced_image)
            self.__get_check_bbox_coords(result, sliced_image.shape[0], slice)   # adds the bbox coordinates to the list wrt orthophoto


    def __raster_to_array_converter(self):
        print("Converting raster to np array (can take a while)....")
        self.dem_array = np.array(self.dataset_dem.ReadAsArray())
        print('Done.')


    def __calculate_slice_bboxes(
        self,
        orthophoto_dataset,
        image_height: int,
        image_width: int,
        slice_height: int = SLICE_SIZE,
        slice_width: int = SLICE_SIZE,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
    ) -> list[list[int]]:
        """
        Given the height and width of an image, calculates how to divide the image into
        overlapping slices according to the height and width provided. These slices are returned
        as bounding boxes in xyxy format.
        :param image_height: Height of the original image.
        :param image_width: Width of the original image.
        :param slice_height: Height of each slice
        :param slice_width: Width of each slice
        :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
        :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
        :return: a list of bounding boxes in xyxy format
        """

        def check_garbage_slice(dataset, xmin, ymin, xmax, ymax):
            
            image = np.array(dataset.ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin))
            image = image[0:3, :, :]
            corner1 = image[:, 0, 0]
            corner2 = image[:, ymax - ymin - 1, xmax - xmin - 1]
            corner3 = image[:, 0, xmax - xmin - 1]
            corner4 = image[:, ymax - ymin - 1, 0]
            reject = np.array([255, 255, 255])
            corners = [corner1, corner2, corner3, corner4]
            for corner in corners:
                if (corner == reject).all():
                    return True
            return False
        
        slice_bboxes = []
        y_max = y_min = 0
        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)
        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    if not check_garbage_slice(orthophoto_dataset, xmin, ymin, xmax, ymax):
                        slice_bboxes.append([xmin, ymin, xmax, ymax])

                else:
                    if not check_garbage_slice(orthophoto_dataset, x_min, y_min, x_max, y_max):
                        slice_bboxes.append([x_min, y_min, x_max, y_max])

                x_min = x_max - x_overlap
            y_min = y_max - y_overlap

        return slice_bboxes

if __name__ == "__main__":
    orthophoto_raster = PATH_TO_ORTHOPHOTO
    dem_raster = PATH_TO_DEM
    pothole_analyzer = PotholeAnalyzer(orthophoto_raster, dem_raster)
    # Analysis result contains the result in the form of (volume, severity_classification, coordinate)
    analysis_result = pothole_analyzer.analyzer()

    print("No potholes found on this road stretch") if analysis_result == None else \
    print ("Process finished!")
    

    
