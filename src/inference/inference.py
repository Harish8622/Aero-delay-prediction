import pandas as pd
from catboost import CatBoostClassifier, Pool
from src.params.inference_params import InferenceParams


params = InferenceParams()
cat_candidates = params.cat_candidates


def predict(
    airline,
    origin,
    destination,
    distance,
    day_of_week,
    month,
    hour_of_day,
    is_bank_holiday,
    dep_rain,
    dep_ice,
    dep_wind,
    arr_rain,
    arr_ice,
    arr_wind,
    model_path="models/catboost_time_split.cbm",
):
    """
    Predict flight delay probability and class.
    All features must be provided explicitly.
    Returns: (probability, prediction)
    """
    input_data = pd.DataFrame(
        [
            {
                "AIRLINE": airline,
                "ORIGIN": origin,
                "DEST": destination,
                "DISTANCE": distance,
                "day_of_week": day_of_week,
                "month": month,
                "hour_of_day": hour_of_day,
                "is_bank_holiday": is_bank_holiday,
                "dep_rain": dep_rain,
                "dep_ice": dep_ice,
                "dep_wind": dep_wind,
                "arr_rain": arr_rain,
                "arr_ice": arr_ice,
                "arr_wind": arr_wind,
            }
        ]
    )
    cat_cols = [c for c in cat_candidates if c in input_data.columns]
    model = CatBoostClassifier()
    print(f"Loading model from {model_path} ...")
    model.load_model(model_path)
    pool = Pool(data=input_data, cat_features=cat_cols)
    proba = model.predict_proba(pool)[:, 1][0]
    threshold = 0.5
    prediction = int(proba >= threshold)
    return proba, prediction


if __name__ == "__main__":
    print("[INFO] Starting tactical inference example...")
    # Tactical (hardcoded) values for all model inputs
    airline = "AA"
    origin = "JFK"
    destination = "LAX"
    distance = 100.0  # Example: JFK-LAX distance in km
    day_of_week = 1  # Tuesday
    month = 7  # July
    hour_of_day = 15  # 3 PM
    is_holiday = 0  # Not a holiday
    dep_rain = 0
    dep_ice = 0
    dep_wind = 0
    arr_rain = 0
    arr_ice = 0
    arr_wind = 0
    print("[INFO] Using tactical values for all features.")
    proba, prediction = predict(
        airline,
        origin,
        destination,
        distance,
        day_of_week,
        month,
        hour_of_day,
        is_holiday,
        dep_rain,
        dep_ice,
        dep_wind,
        arr_rain,
        arr_ice,
        arr_wind,
    )
    print(f"[RESULT] Probability of delay: {proba:.3f}")
    print(f"[RESULT] Predicted delay (1=delay, 0=on time): {prediction}")
