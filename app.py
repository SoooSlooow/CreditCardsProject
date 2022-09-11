import gradio as gr
import pandas as pd
import random

DATA_PATH = 'data/interim/cleaned_dataset.csv'
df = pd.read_csv(DATA_PATH)
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
unique_values = {col: sorted(df[col].unique()) for col in cat_cols}
# print(unique_values['CODE_GENDER'])
# '''

# starting the block

with gr.Blocks() as demo:
    # defining text on the page
    gr.Markdown("""
    **Income Classification with XGBoost 💰**:  This demo uses an XGBoost classifier predicts income based on demographic factors, along with Shapley value-based *explanations*. The [source code for this Gradio demo is here](https://huggingface.co/spaces/gradio/xgboost-income-prediction-with-explainability/blob/main/app.py).
    """)
    # defining the layout
    with gr.Row():
        with gr.Column():
            # defining the inputs
            age = gr.Slider(label='Age', minimum=18, maximum=90, step=1, randomize=True)
            sex = gr.Dropdown(
                label='Sex',
                choices=unique_values['CODE_GENDER'],
                value=lambda: random.choice(unique_values['CODE_GENDER'])
            )
            annual_income = gr.Slider(
                label='Annual income',
                minimum=0,
                maximum=7000000,
                step=10000,
                randomize=True
            )
            income_type = gr.Dropdown(
                label='Income type',
                choices=unique_values['NAME_INCOME_TYPE'],
                value=lambda: random.choice(unique_values['NAME_INCOME_TYPE'])
            )
            work_experience = gr.Slider(
                label='Work experience at current position',
                minimum=0,
                maximum=75,
                step=1,
                randomize=True
            )
            occupation_type = gr.Dropdown(
                label='Occupation type',
                choices=unique_values['OCCUPATION_TYPE'],
                value=lambda: random.choice(unique_values['OCCUPATION_TYPE'])
            )
            education_type = gr.Dropdown(
                label='Education type',
                choices=unique_values['NAME_EDUCATION_TYPE'],
                value=lambda: random.choice(unique_values['NAME_EDUCATION_TYPE'])
            )
            amount_of_family_members = gr.Slider(
                label='Amount of family members',
                minimum=0,
                maximum=12,
                step=1,
                randomize=True
            )
            amount_of_children = gr.Slider(
                label='Amount of children',
                minimum=0,
                maximum=10,
                step=1,
                randomize=True
            )
            family_status = gr.Dropdown(
                label='Family status',
                choices=unique_values['NAME_FAMILY_STATUS'],
                value=lambda: random.choice(unique_values['NAME_FAMILY_STATUS'])
            )
            flag_own_car = gr.Dropdown(
                label='Having a car',
                choices=unique_values['FLAG_OWN_REALTY'],
                value=lambda: random.choice(unique_values['FLAG_OWN_REALTY'])
            )
            flag_own_realty = gr.Dropdown(
                label='Having a realty',
                choices=unique_values['FLAG_OWN_REALTY'],
                value=lambda: random.choice(unique_values['FLAG_OWN_REALTY'])
            )
            housing_type = gr.Dropdown(
                label='Housing type',
                choices=unique_values['NAME_HOUSING_TYPE'],
                value=lambda: random.choice(unique_values['NAME_HOUSING_TYPE'])
            )
            flag_phone = gr.Dropdown(
                label='Having a phone',
                choices=unique_values['FLAG_PHONE'],
                value=lambda: random.choice(unique_values['FLAG_PHONE'])
            )
            flag_work_phone = gr.Dropdown(
                label='Having a work phone',
                choices=unique_values['FLAG_WORK_PHONE'],
                value=lambda: random.choice(unique_values['FLAG_WORK_PHONE'])
            )
            flag_email = gr.Dropdown(
                label='Having an email',
                choices=unique_values['FLAG_EMAIL'],
                value=lambda: random.choice(unique_values['FLAG_EMAIL'])
            )
'''
        with gr.Column():
            # defining the outputs
            label = gr.Label()
            with gr.Row():
                # defining the buttons
                predict_btn = gr.Button(value="Predict")
            # defining the fn that will run when predict is clicked, what it will get as inputs, and which output it will update
            predict_btn.click(
                predict,
                inputs=[
                    age,
                    work_class,
                    education,
                    years,
                    marital_status,
                    occupation,
                    relationship,
                    sex,
                    capital_gain,
                    capital_loss,
                    hours_per_week,
                    country,
                ],
                outputs=[label],
            )
            # defining the fn that will run when interpret is clicked, what it will get as inputs, and which output it will update
            interpret_btn.click(
                interpret,
                inputs=[
                    age,
                    work_class,
                    education,
                    years,
                    marital_status,
                    occupation,
                    relationship,
                    sex,
                    capital_gain,
                    capital_loss,
                    hours_per_week,
                    country,
                ],
                outputs=[plot],
            )
'''

# launch
demo.launch()

