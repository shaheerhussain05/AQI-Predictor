"""
AQI Alert and Health Risk Assessment Module for Pearls AQI Predictor.
Implements US EPA AQI categorization, threshold evaluations, and health advisories.
"""

from typing import Dict, Any, List
from src.config import AQI_CATEGORIES


def get_aqi_category(aqi_value: float) -> Dict[str, Any]:
    """
    Map an AQI numerical value to its official EPA health category.
    """
    aqi_round = round(aqi_value)
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_round <= cat["max"]:
            return {
                "aqi": aqi_round,
                "category": cat["name"],
                "color": cat["color"],
                "icon": cat["icon"],
                "description": cat["description"],
                "advisory": cat["advisory"],
                "is_hazardous": aqi_round > 200,
                "is_unhealthy": aqi_round > 150,
            }
    # Fallback for extreme cases > 1000
    last_cat = AQI_CATEGORIES[-1]
    return {
        "aqi": aqi_round,
        "category": last_cat["name"],
        "color": last_cat["color"],
        "icon": last_cat["icon"],
        "description": last_cat["description"],
        "advisory": last_cat["advisory"],
        "is_hazardous": True,
        "is_unhealthy": True,
    }


def generate_forecast_alerts(forecast_values: List[float]) -> List[Dict[str, Any]]:
    """
    Generate category metadata and alerts for multi-horizon forecast values (Day 1, Day 2, Day 3).
    """
    alerts = []
    for day_idx, val in enumerate(forecast_values, start=1):
        info = get_aqi_category(val)
        info["day"] = f"Day {day_idx} (+{day_idx*24}h)"
        alerts.append(info)
    return alerts


def get_general_recommendations(max_aqi: float) -> List[str]:
    """
    Produce specific checklist recommendations according to peak forecasted AQI.
    """
    recs = []
    if max_aqi <= 50:
        recs.append("Open windows to allow fresh air circulation.")
        recs.append("Ideal conditions for outdoor sports, cycling, and jogging.")
    elif max_aqi <= 100:
        recs.append("Air quality is acceptable for the general population.")
        recs.append("Sensitive individuals should watch for cough or shortness of breath.")
    elif max_aqi <= 150:
        recs.append("Sensitive groups should reduce heavy outdoor exertion.")
        recs.append("Keep windows closed during peak traffic hours.")
        recs.append("Consider running a HEPA air purifier indoors.")
    elif max_aqi <= 200:
        recs.append("Wear a certified N95 / KN95 mask if outdoors.")
        recs.append("Avoid vigorous outdoor cardio or athletic training.")
        recs.append("Keep all residential windows and doors closed.")
        recs.append("Operate air purifiers on high setting.")
    else:
        recs.append("EMERGENCY AIR QUALITY ALERT: Stay indoors as much as possible.")
        recs.append("Avoid ALL outdoor physical activity.")
        recs.append("Run continuous HEPA filtration in sealed living spaces.")
        recs.append("High risk for children, elderly, and respiratory patients.")

    return recs
