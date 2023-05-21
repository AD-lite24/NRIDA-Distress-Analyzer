import numpy as np
from osgeo import gdal, osr
import rasterio as rio
import torch
from tqdm import tqdm

SLICE_SIZE = 512
MM_TO_PIXEL = 3 
PATH_TO_MODEL = 'best.pt'

class PotholeAnalyzer():

    def __init__(self, orthophoto_raster, dem_raster) -> None:
        self.dataset_orthophoto = gdal.Open(orthophoto_raster)
        self.dataset_dem = gdal.Open(dem_raster)
        self.orthophoto_array = None
        self.dem_array = None
        self.slices = None
        self.final_bboxes_ortho = None
        self.final_bboxes_dem = None

    def analyzer(self):

        self.__raster_to_array_converter()
        self.slices = self.__calculate_slice_bboxes(
            image=self.orthophoto_array,
            image_height=self.orthophoto_array[2],
            image_width=self.orthophoto_array[1]
            )
        
        print("Number of minimized slices is ", len(self.slices))

        self.__detector()


    def __get_check_bbox_coords(self, result, image_dim, slice):

        xmin_slice, ymin_slice, xmax_slice, ymax_slice = slice

        boxes = result.xyxyn[0]
        for box in boxes:
            box = np.array(box)
            # print(box)
            xmin, ymin, xmax, ymax, confidence, label = box

            if label == 7:          # index label of the pothole in the in the model

                print("Found!")
                print("before scaling: ", xmin, ymin, xmax, ymax)
                xmin *= image_dim
                xmax *= image_dim
                ymin *= image_dim
                ymax *= image_dim

                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                print("after scaling: ", xmin, ymin, xmax, ymax)

                # Get coordinates wrt original orthophoto
                xmin += xmin_slice
                ymin += ymin_slice
                xmax += xmin_slice
                ymax += ymin_slice

                bbox = (xmin, ymin, xmax, ymax)
                self.final_bboxes.append(bbox)

            else:
                continue

    def __detector(self):

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH_TO_MODEL, source='github')
        for slice in self.slices:
            xmin, ymin, xmax, ymax = slice
            sliced_image = self.orthophoto_array[:, xmin:xmax, ymin:ymax]
            result = model(sliced_image)
            self.__get_check_bbox_coords(result, sliced_image.shape[0], slice)       # adds the bbox coordinates to the list wrt orthophoto

    def __raster_to_array_converter(self):
        self.orthophoto_array = np.array(self.dataset_orthophoto.ReadAsArray())
        self.dem_array = np.array(self.dataset_dem.ReadAsArray())
        self.orthophoto_array = self.orthophoto_array[0:3, :, :] #remove the alpha band

    def __calculate_slice_bboxes(
        self,
        image,
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

        def check_garbage_slice(image, xmin, ymin, xmax, ymax):
    
            corner1 = image[:, xmin-1, ymin-1]
            corner2 = image[:, xmin-1, ymax-1]
            corner3 = image[:, xmax-1, ymin-1]
            corner4 = image[:, xmax-1, ymax-1]
            reject = [255, 255, 255]
            corners = [corner1, corner2, corner3, corner4]

            for corner in corners:
                if (corner == reject).all():
                    return True

            return False
        
        print(image.shape)
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
                    if not check_garbage_slice(image, xmin, ymin, xmax, ymax):
                        slice_bboxes.append([xmin, ymin, xmax, ymax])

                else:
                    if not check_garbage_slice(image, x_min, y_min, x_max, y_max):
                        slice_bboxes.append([x_min, y_min, x_max, y_max])

                x_min = x_max - x_overlap
            y_min = y_max - y_overlap

        return slice_bboxes


    

if __name__ == "__main__":
    orthophoto_raster, dem_raster = ''
    pothole_analyser = PotholeAnalyzer(orthophoto_raster, dem_raster)
    pothole_analyser.analyzer()
