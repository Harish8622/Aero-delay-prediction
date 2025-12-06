class InferenceParams:
    def __init__(self):
        self.cat_candidates = ["AIRLINE", "ORIGIN", "DEST"]
        self.num_candidates = [
            "DISTANCE",
            "day_of_week",
            "month",
            "hour_of_day",
            "is_bank_holiday",
            "dep_rain",
            "dep_ice",
            "dep_wind",
            "arr_rain",
            "arr_ice",
            "arr_wind",
        ]
