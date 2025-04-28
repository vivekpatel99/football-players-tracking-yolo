import json
import logging
import os
import sys
from calendar import c
from pathlib import Path

import hydra
import numpy as np
import pyrootutils
import supervision as sv
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = ""


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="inference.yaml")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    INPUT_VIDEO = cfg.paths.ip_vid_path
    log.info(f"input video: {INPUT_VIDEO}")
    output_file = Path(cfg.paths.processed_vid_path)
    log.info(f"output video: {output_file}")
    output_file.unlink(missing_ok=True)  # Remove existing file if it exists

    # --- Annotator setup ---
    log.info("annotator setup")
    label_annotator = hydra.utils.instantiate(cfg.annotate.label_annotator, text_position=sv.Position.TOP_CENTER)
    traingle_annotator = hydra.utils.instantiate(cfg.annotate.traingle_annotator)
    ellipse_annotator = hydra.utils.instantiate(cfg.annotate.ellipse_annotator)

    # --- Models instantiation ---
    log.info(f"{cfg.models.model_name} instantiation")
    model = hydra.utils.instantiate(cfg.models.model)
    log.info("team classifier instantiation")
    team_classifier = hydra.utils.instantiate(cfg.models.team_classifier)
    log.info(f"tracker instantiation: {cfg.tracker.tracker}")

    classes_dict = {key: val for val, key in enumerate(cfg.datasets.names)}
    PLAYER_ID = classes_dict["player"]
    BALL_ID = classes_dict["ball"]
    GOALKEEPER_ID = classes_dict["goalkeeper"]
    REFEREE_ID = classes_dict["referee"]
    FINAL_CLASS_ID_GOALKEEPER = 0
    # FINAL_CLASS_ID_TEAM_1 = 1
    # FINAL_CLASS_ID_TEAM_2 = 2
    FINAL_CLASS_ID_REFEREE = 3

    log.info("collecting crops")
    pred_args = hydra.utils.instantiate(cfg.args, _convert_="object")
    frame_generator = sv.get_video_frames_generator(source_path=INPUT_VIDEO, stride=30)
    crops = []
    for frame in frame_generator:
        result = model.predict(frame, **pred_args)[0]
        detections = sv.Detections.from_ultralytics(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        crops += players_crops

    log.info("training classifier")
    team_classifier.fit(crops)

    log.info("processing video")

    def separate_teams_callback(frame: np.ndarray, _) -> np.ndarray:
        # _frame = frame.copy()

        # --- Object Detection ---
        result = model.track(frame, persist=True, tracker=hydra.utils.to_absolute_path(cfg.tracker.tracker), **pred_args)[0]
        detections = sv.Detections.from_ultralytics(result)

        # --- Ball Detection ---
        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # --- Non-Ball Detections & Tracking ---
        non_ball_detections = detections[detections.class_id != BALL_ID]
        non_ball_detections = non_ball_detections.with_nms(
            threshold=0.5,
            class_agnostic=False,  # Usually better to do NMS per class
        )
        # --- Separate Detections by Original Class ---
        goalkeepers_detections = non_ball_detections[non_ball_detections.class_id == GOALKEEPER_ID]
        players_detections = non_ball_detections[non_ball_detections.class_id == PLAYER_ID]
        referees_detections = non_ball_detections[non_ball_detections.class_id == REFEREE_ID]

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        # --- Assign Final Class IDs ---
        # Goalkeepers get final ID 1
        goalkeepers_detections.class_id = np.full(len(goalkeepers_detections), FINAL_CLASS_ID_GOALKEEPER)
        # Referees get final ID 4
        referees_detections.class_id = np.full(len(referees_detections), FINAL_CLASS_ID_REFEREE)

        # --- Merge all detections for annotation --
        all_detections = sv.Detections.merge([goalkeepers_detections, players_detections, referees_detections])
        # Ensure class_id is integer type for palette indexing
        all_detections.class_id = all_detections.class_id.astype(int)

        # --- Create Labels (Using Tracker ID) ---
        labels = [f"#{tid}" for tid in all_detections.tracker_id]

        # --- Annotation ---
        # Annotate non-ball detections (Goalkeepers, Players, Referees) with ellipses
        annot_frame = ellipse_annotator.annotate(scene=frame, detections=all_detections)

        # Annotate the ball separately
        annot_frame = traingle_annotator.annotate(scene=annot_frame, detections=ball_detections)

        # Add labels to non-ball detections
        annot_frame = label_annotator.annotate(scene=annot_frame, detections=all_detections, labels=labels)
        return annot_frame

    sv.process_video(
        source_path=INPUT_VIDEO,
        target_path=str(output_file),
        callback=separate_teams_callback,
    )


if __name__ == "__main__":
    main()
#     BALL_ID = 0
#     colors_list = [sv.Color.RED, sv.Color.WHITE, sv.Color.GREEN, sv.Color.BLUE]
#     colors = sv.ColorPalette(colors=colors_list)

#     # ellip_annotator = sv.EllipseAnnotator(color=colors,thickness=2)
#     ellip_annotator = sv.EllipseAnnotator(color=colors, thickness=2)
#     traingle_annot = sv.TriangleAnnotator(color=colors_list[0], base=25, height=21, outline_thickness=1)
#     label_annotator = sv.LabelAnnotator(color=colors, text_color=sv.Color.BLACK, text_position=sv.Position.TOP_CENTER)
#     from ultralytics import YOLO

#     model = YOLO("/workspaces/football-players-tracking-yolo/results/augumented-data-yolo12l/weights/final_best.pt")

#     def yolo_tracker_callback(frame: np.ndarray, _: int) -> np.ndarray:
#         # https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies
#         result = model.track(frame, show=False)[0]
#         # result = model.predict(frame, imgsz=pred_args["imgsz"], conf=0.3)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         ball_detections = detections[detections.class_id == BALL_ID]
#         ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

#         rest_detections = detections[detections.class_id != BALL_ID]
#         rest_detections = rest_detections.with_nms(threshold=0.5, class_agnostic=True)
#         rest_detections.class_id -= 1

#         labels = [f"#{id} {confidence:.2f}" for id, confidence in zip(detections.tracker_id, detections.confidence, strict=False)]
#         annot_frame = ellip_annotator.annotate(scene=frame.copy(), detections=rest_detections)
#         annot_frame = label_annotator.annotate(scene=annot_frame, detections=rest_detections, labels=labels)
#         annot_frame = traingle_annot.annotate(scene=annot_frame, detections=ball_detections)
#         return annot_frame


# sv.process_video(
#     source_path="/workspaces/football-players-tracking-yolo/data/121364_0.mp4",
#     target_path="/workspaces/football-players-tracking-yolo/results/yolo_tracker_result.mp4",
#     callback=yolo_tracker_callback,
# )
