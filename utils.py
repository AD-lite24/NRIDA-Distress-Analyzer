import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from osgeo import gdal, osr

def show_image(image, bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((y_min, x_max), x_max - x_min,
                                 y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig("test.png")


def show_image_2(image):

    r, g, b = image[0], image[1], image[2]
    rgb_image = np.stack([r, g, b], axis=-1)
    plt.imshow(rgb_image)
    plt.savefig("test1.png")

# convert the final bbox coords wrt to the ones on the DEM
def coord_converter(xmin_photo, ymin_photo, xmax_photo, ymax_photo, dataset_photo, dataset_dem):

    # Get geotransform of the orthophoto
    geotransform_photo = dataset_photo.GetGeoTransform()

    # Get the real world coordinates
    x_min_real = geotransform_photo[0] + xmin_photo * \
        geotransform_photo[1] + ymin_photo * geotransform_photo[2]
    y_min_real = geotransform_photo[3] + xmin_photo * \
        geotransform_photo[4] + ymin_photo * geotransform_photo[5]

    x_max_real = geotransform_photo[0] + xmax_photo * \
        geotransform_photo[1] + ymax_photo * geotransform_photo[2]
    y_max_real = geotransform_photo[3] + xmax_photo * \
        geotransform_photo[4] + ymax_photo * geotransform_photo[5]

    # use real word coords on the geotagged orthophoto
    geotransform_photo_dem = dataset_dem.GetGeoTransform()

    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset_dem.GetProjection())

    coord_transform = osr.CoordinateTransformation(srs, srs.CloneGeogCS())

    # Convert the real-world coordinates to projected coordinates
    x_proj_min, y_proj_min, _ = coord_transform.TransformPoint(
        x_min_real, y_min_real)
    x_proj_max, y_proj_max, _ = coord_transform.TransformPoint(
        x_max_real, y_max_real)

    # Apply the geotransform_photoation to obtain pixel coordinates
    x_min_dem = int((x_proj_min - geotransform_photo_dem[0]) / geotransform_photo_dem[1])
    y_min_dem = int((y_proj_min - geotransform_photo_dem[3]) / geotransform_photo_dem[5])

    x_max_dem = int((x_proj_max - geotransform_photo_dem[0]) / geotransform_photo_dem[1])
    y_max_dem = int((y_proj_max - geotransform_photo_dem[3]) / geotransform_photo_dem[5])

    # Note the last two items are the real world coordinates of the centroid of the bbox to identify and locate each pothole
    return x_min_dem, y_min_dem, x_max_dem, y_max_dem, (x_min_real + x_max_real)/2, (y_min_real + y_max_real)/2
