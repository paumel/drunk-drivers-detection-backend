import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import scipy.spatial as spatial
import numpy as np


def prepareTF():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1
    # Patch the location of gfile
    tf.gfile = tf.io.gfile


def getCategoryIndex():
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'data/train.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True)
    return category_index


def load_model():
    model_dir = "exported_model/saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'],
            output_dict['detection_boxes'],
            image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(
            detection_masks_reframed > 0.5,
            tf.uint8
        )

        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def do_kdtree(combined_x_y_arrays, points):
    mytree = spatial.cKDTree(combined_x_y_arrays)
    return mytree.query(points)


def lastNumberOfPoints(point, frames, framesIndex, number=10):
    count = 1
    points = []
    currentPoint = point
    while framesIndex >= count and number != count:
        if len(frames[framesIndex-count]) > 0:
            dist, indexes = do_kdtree(frames[framesIndex-count], [currentPoint])
            if dist[0] < 100:
                points.append(frames[framesIndex-count][indexes[0]])
                currentPoint = frames[framesIndex-count][indexes[0]]
        count += 1
    return points


def getCurrentFramePoints(output_dict, MIN_SCORE, height, width):
    boxesIndex = 0
    currentFrame = []
    while output_dict['detection_scores'][boxesIndex] > MIN_SCORE:
        box = output_dict['detection_boxes'][boxesIndex]
        (ymin, xmin, ymax, xmax) = (box[0]*height, box[1]*width, box[2]*height, box[3]*width)
        (x_avg, y_avg) = ((xmin+xmax)/2, (ymin+ymax)/2)
        currentFrame.append([x_avg, y_avg])
        boxesIndex += 1
    return currentFrame
