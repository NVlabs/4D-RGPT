from typing import Any, List, Dict, Optional
import pandas as pd
import numpy as np

def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05
) -> float:
    """
    `https://github.com/VITA-Group/VLM-3R/blob/main/thinking-in-space/lmms_eval/tasks/vstibench/utils.py`
    """
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    try:
        abs_dist_norm = abs(pred - target) / target
    except TypeError:
        return 0.0
    accuracy = abs_dist_norm <= 1 - conf_intervs
    return accuracy.mean()

# tolerance:
#   direction: # +/- 10 degrees of the gt direction
#     range: 10
#     fn: angular_absolute

#   distance: # 25 cm when the gt is 1 m
#     range: 0.25
#     fn: relative

#   velocity: # +/- 2 m/s when the gt is 10 m/s
#     range: 0.2
#     fn: relative
#     absolute_floor:
#       threshold: 0.5  # Switch to absolute tolerance if gt is below 1.0 m/s


def numeric_success(pred: float, gt: float, tolerance: Dict) -> bool:
    fn = tolerance['fn']
    match fn:
        case "relative":
            return relative_success(pred, gt, tolerance)
        # case "absolute":
        #     return absolute_success(pred, gt, tolerance)
        # case "exact":
        #     return exact_match_success(pred, gt, tolerance)
        case "angular_absolute":
            return angular_absolute_success(pred, gt, tolerance)
        case _:
            raise ValueError(f"Unsupported judge function: {fn}")

def relative_success(pred: float, gt: float, tolerance: dict) -> bool:
    if not isinstance(pred, float):
        # return {"success": False, "reason": "invalid_prediction"}
        return False

    relative_range = tolerance["range"]

    # Check if a hybrid absolute tolerance floor is defined and the GT is below the threshold
    if (
        "absolute_floor" in tolerance
        and gt < tolerance["absolute_floor"]["threshold"]
    ):
        # Use a fixed absolute tolerance calculated AT the threshold
        threshold = tolerance["absolute_floor"]["threshold"]
        abs_tolerance = threshold * relative_range  # Dynamic calculation

        success = abs(pred - gt) <= abs_tolerance
        min_value = gt - abs_tolerance
        max_value = gt + abs_tolerance
        fn_used = "absolute_floor"
    else:
        # Default to relative tolerance for values above the threshold
        min_value = gt * (1 - relative_range)
        max_value = gt * (1 + relative_range)
        success = min_value <= pred <= max_value
        fn_used = "relative"

    # return {
    #     "success": success,
    #     "min_value": min_value,
    #     "max_value": max_value,
    #     "pred": pred,
    #     "gt": gt,
    #     "fn_used": fn_used,
    # }
    return success

# def absolute_success(pred: float, gt: float, tolerance: dict) -> bool:
#     if not isinstance(pred, float):
#         # return {"success": False, "reason": "invalid_prediction"}
#         return False

#     tolerance_range = tolerance["range"]
#     success = abs(pred - gt) <= tolerance_range
#     # return {
#     #     "success": success,
#     #     "tolerance_range": tolerance_range,
#     #     "diff": abs(pred - gt),
#     #     "pred": pred,
#     #     "gt": gt,
#     # }
#     return success

def angular_absolute_success(pred: Optional[float], gt: float, tolerance_config: dict) -> bool:
    # if not isinstance(pred, float):
    #     # return {"success": False, "reason": "invalid_prediction"}
    #     return False
    if pred is None:
        return False
    if gt is None:
        return True

    tolerance_range = tolerance_config["range"]
    # Calculate angular difference considering wraparound (-180 to 180)
    diff = (pred - gt + 180) % 360 - 180
    success = abs(diff) <= tolerance_range
    # return {
    #     "success": success,
    #     "tolerance_range": tolerance_range,
    #     "diff": abs(diff),
    #     "pred": pred,
    #     "gt": gt,
    # }
    return success

# def exact_match_success(pred: float, gt: float, tolerance_config: dict) -> dict:
#     if not isinstance(pred, float):
#         # return {"success": False, "reason": "invalid_prediction"}
#         return False

#     success = pred == gt
#     # return {
#     #     "success": success,
#     #     "pred": pred,
#     #     "gt": gt,
#     # }
#     return success


def judge_localization_success(
    distance_gt: float,
    direction_gt: float,
    distance_pred: Optional[float],
    direction_pred: Optional[float],
    tolerance_config: Dict,
) -> dict:
    """
    Judge localization success for polar coordinate position evaluation.
    This function evaluates whether a predicted position falls within the acceptable region.

    Args:
        distance_gt (Real): Ground truth distance in meters
        direction_gt (Real): Ground truth direction in degrees (-180 to 180)
        distance_pred (Real | None): Predicted distance in meters
        direction_pred (Real | None): Predicted direction in degrees
        tolerance_config (dict): Configuration containing tolerance settings for distance and direction

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether both distance and direction are within tolerance
            - distance_success (bool): Whether distance is within tolerance
            - direction_success (bool): Whether direction is within tolerance
            - distance_details (dict): Detailed distance evaluation results
            - direction_details (dict): Detailed direction evaluation results

    """
    # Get tolerance configurations for each component
    distance_tolerance = tolerance_config.get("distance", {})
    direction_tolerance = tolerance_config.get("direction", {})

    # Evaluate distance and direction by calling the generic judge_success function
    distance_success = numeric_success(
        distance_gt, distance_pred, distance_tolerance)
    # print(direction_gt, direction_pred)
    direction_success = numeric_success(
        direction_gt, direction_pred, direction_tolerance)

    # distance_success = distance_details.get("success", False)
    # direction_success = direction_details.get("success", False)

    # LSR success requires both to be within tolerance
    lsr_success = distance_success and direction_success

    return lsr_success
# {
#         "success": lsr_success,
#         "distance_success": distance_success,
#         "direction_success": direction_success,
#         "distance_details": distance_details,
#         "direction_details": direction_details,
#     }




def localization_success_rate(
    # raw_results_list: List[Dict[str, Any]],
    result: Dict[str, Any],
    config: Dict[str, Any],
    annotation_dict: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Calculate Localization Success Rate (LSR) for distance-direction pairs.

    Args:
        raw_results_list: List of individual evaluation results
        config: Configuration dictionary containing tolerance settings

    Returns:
        List of LSR evaluation results
    """
    # lsr_results: list[dict[str, Any]] = []

    # # Group results by image_group, time, and ego_or_target
    # # This allows pairing distance and direction questions from the same image group
    # grouped_data: dict[str, dict] = {}

    # for result in raw_results_list:
    # question_id = result["question_id"]
    # image_group = result["group_id"]
    # # image_group = extract_group_id(question_id)
    # time = result["time"]
    # ego_or_target = result["ego_or_target"]
    # qa_type = result["qa_type"]

    # # Create a unique key for grouping based on image_group, time, and ego_or_target
    # # This allows pairing distance and direction from the same image group
    # group_key = f"{image_group}_{time}_{ego_or_target}"

    # if group_key not in grouped_data:
    #     grouped_data[group_key] = {
    #         "image_group": image_group,
    #         "time": time,
    #         "ego_or_target": ego_or_target,
    #         "distance": None,
    #         "direction": None,
    #     }

    # if qa_type == "distance":
    #     grouped_data[group_key]["distance"] = result
    # elif qa_type == "direction":
    #     grouped_data[group_key]["direction"] = result

    # Calculate LSR for groups - now handle cases where either distance or direction is missing
    # for group_data in result:
    if len(result["answer"]) != 2:
        return None
    distance_result, direction_result = result["answer"]

    # # Skip only if both distance and direction are missing (no data for this group)
    # if distance_result is None and direction_result is None:
    #     continue

    # Determine which result to use as primary source for metadata
    primary_result = distance_result if distance_result is not None else direction_result

    # Extract or set default values for LSR calculation
    distance_gt = distance_result["gt_value"] if distance_result is not None else None
    direction_gt = direction_result["gt_value"] if direction_result is not None else None
    distance_pred = distance_result["pred_value"] if distance_result is not None else None
    direction_pred = direction_result["pred_value"] if direction_result is not None else None

    # Calculate LSR using the metric_utils function
    lsr_details = judge_localization_success(
        distance_gt=distance_gt,
        direction_gt=direction_gt,
        distance_pred=distance_pred,
        direction_pred=direction_pred,
        tolerance_config=config["tolerance"],
    )

    # Use the qa_category from primary result for LSR
    primary_qa_category = primary_result["qa_category"]

    # Get annotation information from primary result
    group_id = primary_result.get("group_id", "")
    object_type = primary_result.get("object_type")
    relation = primary_result.get("relation")

    # Create LSR result entry using primary question_id
    lsr_result = {
        "question_id": primary_result["question_id"],  # Use primary question_id
        "group_id": group_id,
        "object_type": object_type,
        "relation": relation,
        "qa_category": primary_qa_category,
        "time": group_data["time"],
        "ego_or_target": group_data["ego_or_target"],
        "qa_type": "lsr",
        "gt_value": {"distance": distance_gt, "direction": direction_gt},
        "pred_value": {"distance": distance_pred, "direction": direction_pred},
        "gt_unit": {
            "distance": distance_result["gt_unit"] if distance_result is not None else None,
            "direction": direction_result["gt_unit"] if direction_result is not None else None,
        },
        "pred_unit": {
            "distance": distance_result["pred_unit"] if distance_result is not None else None,
            "direction": direction_result["pred_unit"]
            if direction_result is not None
            else None,
        },
        "success": lsr_details["success"],
        "details": lsr_details,
    }

    # lsr_results.append(lsr_result)

    # return lsr_results


def compute_tlc_mlsr(
    df: pd.DataFrame,
    *,
    max_step: int = 3,
) -> pd.DataFrame:
    """Compute TLC@k (k=1..max_step), TLC, MLSR

    Return DataFrame columns:
        relation | TC@1 | ... | TLC@max_step | TLC | MLSR

    - TLC@k : The proportion of cases where success is achieved at t=0 and then consecutively maintained for frames 1 through k.
    - TLC   : The proportion of cases where success is consecutively maintained across all frames (Temporal Localization Consistency).
    - MLSR  : For each sequence, compute "successful frames / (max_step + 1)" and then take the average per relation.

    """
    lsr = df[df.qa_type == "lsr"].copy()

    pivot = (
        lsr.pivot_table(
            index=["group_id", "relation"], columns="time", values="success", aggfunc="first"
        )
        .fillna(False)
        .astype(bool)
    )

    frames = []
    for k in range(1, max_step + 1):
        seq_ok = pivot[0] & pivot.loc[:, list(range(1, k + 1))].all(axis=1)
        rel_df = (
            seq_ok.reset_index(name="ok")
            .groupby("relation")["ok"]
            .mean()
            .reset_index(name=f"TLC@{k}")
        )
        overall = pd.DataFrame([{"relation": "Overall", f"TLC@{k}": seq_ok.mean()}])
        frames.append(pd.concat([overall, rel_df], ignore_index=True))

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="relation", how="outer")

    strict_ok = pivot.loc[:, list(range(0, max_step + 1))].all(axis=1)
    strict_df = (
        strict_ok.reset_index(name="ok").groupby("relation")["ok"].mean().reset_index(name="TLC")
    )
    strict_overall = pd.DataFrame([{"relation": "Overall", "TLC": strict_ok.mean()}])
    strict_df = pd.concat([strict_overall, strict_df], ignore_index=True)
    merged = merged.merge(strict_df, on="relation", how="outer")

    mf_score = pivot.mean(axis=1)
    mf_df = (
        mf_score.reset_index(name="score")
        .groupby("relation")["score"]
        .mean()
        .reset_index(name="MLSR")
    )
    mf_overall = pd.DataFrame([{"relation": "Overall", "MLSR": mf_score.mean()}])
    mf_df = pd.concat([mf_overall, mf_df], ignore_index=True)
    merged = merged.merge(mf_df, on="relation", how="outer")

    merged = pd.concat(
        [
            merged.loc[merged.relation == "Overall"],
            merged.loc[merged.relation != "Overall"].sort_values("TLC", ascending=False),
        ]
    ).reset_index(drop=True)

    return merged


def aggregate_results(raw_results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the aggregated results in dictionary format.
    TLC / MLSR are also calculated here.
    """
    # jst = timezone(timedelta(hours=9))
    # evaluated_at = datetime.now(jst).isoformat()
    result_dict: dict[str, Any] = {"evaluated_at": evaluated_at}

    # Success rates (each frame independently)
    success_rates = calculate_success_rate(raw_results_list)
    result_dict.update(success_rates)

    # Temporal consistency metrics
    df_all = pd.DataFrame(raw_results_list)
    df_lsr = df_all[df_all.qa_type == "lsr"]
    df_tlc_mlsr = compute_tlc_mlsr(df_lsr)
    result_dict["temporal_consistency"] = df_tlc_mlsr.to_dict(orient="records")

    return result_dict