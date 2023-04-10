import numpy as np
import cv2

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

def decode_predictions(scores, geometry, score_threshold, nms_threshold):
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data_0 = geometry[0, 0, y]
        x_data_1 = geometry[0, 1, y]
        x_data_2 = geometry[0, 2, y]
        x_data_3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            if scores_data[x] < score_threshold:
                continue

            (offset_x, offset_y) = (x * 4.0, y * 4.0)
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data_0[x] + x_data_2[x]
            w = x_data_1[x] + x_data_3[x]

            end_x = int(offset_x + (cos * x_data_1[x]) + (sin * x_data_2[x]))
            end_y = int(offset_y - (sin * x_data_1[x]) + (cos * x_data_2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    boxes = np.array(rects)
    boxes = non_max_suppression(boxes, probs=confidences, overlapThresh=nms_threshold)

    return boxes

def extract_bottom_right_subimage(image, rects, rW, rH, padding_ratio=0.2):
    if rects.size == 0:
        return None

    # Triez les rectangles en fonction de la position (end_y, end_x) en ordre décroissant
    rects = sorted(rects, key=lambda x: (x[3], x[2]), reverse=True)
    bottom_right_rect = rects[0]

    start_x, start_y, end_x, end_y = bottom_right_rect
    width = int((end_x - start_x) * rW)
    height = int((end_y - start_y) * rH)

    padding_w = int(width * padding_ratio)
    padding_h = int(height * padding_ratio)

    start_x = int(start_x * rW) - padding_w
    start_y = int(start_y * rH) - padding_h
    end_x = int(end_x * rW) + padding_w
    end_y = int(end_y * rH) + padding_h

    # Limitez les coordonnées pour éviter les débordements
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(image.shape[1], end_x)
    end_y = min(image.shape[0], end_y)

    return image[start_y:end_y, start_x:end_x]


def detect_text_regions(f, net, layerNames, newW, newH,debug=False):
    (H, W) = f.shape[:2]
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(f, (newW, newH))
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                  (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    rectangles = decode_predictions(scores, geometry, 0.5, 0.3)

    if len(rectangles) == 0:
        print("Aucun rectangle détecté")
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    else:
        print("Rectangle dectecté")
        nf = extract_bottom_right_subimage(f, rectangles, rW, rH)
        if nf is None:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(nf, cv2.COLOR_BGR2GRAY)

    return gray


