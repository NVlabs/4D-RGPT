import numpy as np
import supervision as sv

def mask_to_bbox(mask: np.ndarray, format='xyxy') -> tuple[int, int, int, int]:
    assert(format == "xyxy")
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return (xmin, ymin, xmax, ymax)

def apply_set_of_marks(
    cv2_image: np.ndarray,
    labels : list[str],
    input_boxes: np.ndarray | None = None,
    masks: np.ndarray | None = None
) -> np.ndarray:
    """
    """
    class_ids = np.array(list(range(len(labels))))

    annotated_frame = cv2_image.copy()

    assert(masks is not None or input_boxes is not None)

    if input_boxes is None:
        input_boxes = np.array([ mask_to_bbox(mask) for mask in masks ])

    if masks is None:
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

    else:
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame


def apply_set_of_marks_v2(
    cv2_image: np.ndarray,
    labels : list[str],
    input_boxes: np.ndarray | None = None,
    masks: np.ndarray | None = None
) -> np.array:
    MIN_AREA_PERCENTAGE = 0.005
    MAX_AREA_PERCENTAGE = 0.05

    # load image
    image_bgr = cv2_image

    # segment image
    # detections = sv.Detections.from_sam(sam_result=sam_result)

    class_ids = np.array(list(range(len(labels))))
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    # filter masks
    height, width, channels = image_bgr.shape
    image_area = height * width

    min_area_mask = (detections.area / image_area) > MIN_AREA_PERCENTAGE
    max_area_mask = (detections.area / image_area) < MAX_AREA_PERCENTAGE
    detections = detections[min_area_mask & max_area_mask]

    # setup annotators
    mask_annotator = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.3
    )
    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_position=sv.Position.CENTER,
        text_scale=1,
        text_color=sv.Color.white(),
        color=sv.Color.black(),
        text_thickness=2
    )

    # annotate
    labels = [str(i) for i in range(len(detections))]

    annotated_image = mask_annotator.annotate(
        scene=image_bgr.copy(), detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    # sv.plot_image(annotated_image)
    return annotated_image




def apply_time_marks(
    cv2_image: np.ndarray,
    time: float
) -> np.ndarray:
    """
    Apply time marks to cv2_image
    """
    annotated_frame = cv2_image.copy()

    height, width = annotated_frame.shape[:2]

    # box_size = [0.05, 0.10]

    start_x = int(width * 0.01)
    end_x = int(width * 0.05)

    start_y = int(height * 0.08)
    end_y = int(height * 0.12)

    boxes = np.array([[start_x, start_y, end_x, end_y]])

    detections = sv.Detections(
        xyxy=boxes,  # (n, 4)
        # mask=masks.astype(bool),  # (n, h, w)
        class_id=np.array([0])
    )

    box_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=3, text_position=sv.Position.BOTTOM_RIGHT)
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=[f"{time:.1f} s"],
    )

    return annotated_frame


def draw_som_on_image(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    labels: list[int],
    label_mode: str = "1",
    alpha: float = 0.1,
    anno_mode: list[str] = ["Mask"],
) -> np.ndarray:
    """
    Draw SoM on image.

    Args:
        image_rgb (np.ndarray): image to draw on
        masks (list[np.ndarray]): masks to draw
        labels (list[int]): labels to draw
        label_mode (str)
        alpha (float)
        anno_mode (list[str]): Annotation types to draw

    Returns:
        np.ndarray, image with SoM drawn on it
    """
    raise NotImplementedError()

    if len(masks) == 0:
        raise ValueError("No masks provided to draw SoM on image")

    if len(masks) != len(labels):
        raise ValueError(
            f"Number of masks and labels must be the same: masks={len(masks)}, labels={len(labels)}"
        )

    visual = Visualizer(image_rgb, metadata=metadata)
    for i in range(len(masks)):
        label = labels[i]
        mask = masks[i]
        demo = visual.draw_binary_mask_with_number(
            mask,
            text=str(label),
            label_mode=label_mode,
            alpha=alpha,
            anno_mode=anno_mode,
        )
    return demo.get_image()


if __name__ == "__main__":
    import cv2
    image = cv2.imread("demo/frame_00000.jpg")
    output_image = apply_time_marks(image, 12.3)
    cv2.imwrite("demo/frame_0000.time.jpg", output_image)
