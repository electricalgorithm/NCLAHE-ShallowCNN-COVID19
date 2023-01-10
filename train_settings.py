"""
A settings dictionary for training model.
"""

settings = {
    # Dataset directories.
    "DIR_DATASET": "",
    "DIR_COVID": "Class_1_COVID/",
    "DIR_NON_COVID": "Class_0_Normal/",

    # Model settings.
    "MODEL_NAME": "covid_nclahe",
    "MODEL_INPUT_SIZES": (128, 128),
    "MODEL_EPOCH": 10,
    "MODEL_BATCH": 32,
}