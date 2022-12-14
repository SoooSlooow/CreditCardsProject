import subprocess
import random
from typing import Any

import gradio as gr
import joblib
import numpy as np
import pandas as pd

OUTPUT_DATA_PATH = "data/processed/app_dataset.csv"
PREDICTIONS_PATH = "models/predictions/app_predictions.csv"
UNIQUE_VALUES_PATH = "models/other/unique_column_values.pkl"


def predict(*args: tuple) -> Any:
    app_df = pd.DataFrame(data=[args], columns=columns, index=[0])
    app_df.to_csv(OUTPUT_DATA_PATH, index=False)
    subprocess.run(
        [
            "python",
            "-m",
            "src.models.make_predictions",
            "data/processed/app_dataset.csv",
            "models/final_model.pkl",
            "models/predictions/app_predictions.csv",
        ],
        shell=True,
    )
    predictions = np.genfromtxt(PREDICTIONS_PATH, delimiter=",", skip_header=1)
    if predictions[2] == 1:
        message = "Client is considered bad. Issuance of credit is not recommended."
    else:
        message = "Client is considered good. Issuance of credit is allowed."
    return round(predictions[0], 3), message


columns = (
    "YEARS_BIRTH",
    "CODE_GENDER",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "YEARS_EMPLOYED",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "CNT_FAM_MEMBERS",
    "CNT_CHILDREN",
    "NAME_FAMILY_STATUS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_HOUSING_TYPE",
    "FLAG_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_EMAIL",
)
unique_values = joblib.load(UNIQUE_VALUES_PATH)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            age = gr.Slider(label="Age", minimum=18, maximum=90, step=1, randomize=True)
            sex = gr.Dropdown(
                label="Sex",
                choices=unique_values["CODE_GENDER"],
                value=lambda: random.choice(unique_values["CODE_GENDER"]),
            )
            annual_income = gr.Slider(
                label="Annual income",
                minimum=0,
                maximum=1000000,
                step=10000,
                randomize=True,
            )
            income_type = gr.Dropdown(
                label="Income type",
                choices=unique_values["NAME_INCOME_TYPE"],
                value=lambda: random.choice(unique_values["NAME_INCOME_TYPE"]),
            )
            work_experience = gr.Slider(
                label="Work experience at current position",
                minimum=0,
                maximum=75,
                step=1,
                randomize=True,
            )
            occupation_type = gr.Dropdown(
                label="Occupation type",
                choices=unique_values["OCCUPATION_TYPE"],
                value=lambda: random.choice(unique_values["OCCUPATION_TYPE"]),
            )
            education_type = gr.Dropdown(
                label="Education type",
                choices=unique_values["NAME_EDUCATION_TYPE"],
                value=lambda: random.choice(unique_values["NAME_EDUCATION_TYPE"]),
            )
            amount_of_family_members = gr.Slider(
                label="Amount of family members",
                minimum=0,
                maximum=12,
                step=1,
                randomize=True,
            )
            amount_of_children = gr.Slider(
                label="Amount of children",
                minimum=0,
                maximum=10,
                step=1,
                randomize=True,
            )

        with gr.Column():
            family_status = gr.Dropdown(
                label="Family status",
                choices=unique_values["NAME_FAMILY_STATUS"],
                value=lambda: random.choice(unique_values["NAME_FAMILY_STATUS"]),
            )
            flag_own_car = gr.Dropdown(
                label="Having a car",
                choices=unique_values["FLAG_OWN_REALTY"],
                value=lambda: random.choice(unique_values["FLAG_OWN_REALTY"]),
            )
            flag_own_realty = gr.Dropdown(
                label="Having a realty",
                choices=unique_values["FLAG_OWN_REALTY"],
                value=lambda: random.choice(unique_values["FLAG_OWN_REALTY"]),
            )
            housing_type = gr.Dropdown(
                label="Housing type",
                choices=unique_values["NAME_HOUSING_TYPE"],
                value=lambda: random.choice(unique_values["NAME_HOUSING_TYPE"]),
            )
            flag_phone = gr.Dropdown(
                label="Having a phone",
                choices=unique_values["FLAG_PHONE"],
                value=lambda: random.choice(unique_values["FLAG_PHONE"]),
            )
            flag_work_phone = gr.Dropdown(
                label="Having a work phone",
                choices=unique_values["FLAG_WORK_PHONE"],
                value=lambda: random.choice(unique_values["FLAG_WORK_PHONE"]),
            )
            flag_email = gr.Dropdown(
                label="Having an email",
                choices=unique_values["FLAG_EMAIL"],
                value=lambda: random.choice(unique_values["FLAG_EMAIL"]),
            )

        with gr.Column():
            label_1 = gr.Label(label="Client rating")
            label_2 = gr.Textbox(label="Client verdict (client is considered bad if client rating < 0.99)")
            with gr.Row():
                predict_btn = gr.Button(value="Predict")
            predict_btn.click(
                predict,
                inputs=[
                    age,
                    sex,
                    annual_income,
                    income_type,
                    work_experience,
                    occupation_type,
                    education_type,
                    amount_of_family_members,
                    amount_of_children,
                    family_status,
                    flag_own_car,
                    flag_own_realty,
                    housing_type,
                    flag_phone,
                    flag_work_phone,
                    flag_email,
                ],
                outputs=[label_1, label_2],
            )

demo.launch()
