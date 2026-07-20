import os
import gradio as gr
import pandas as pd
import joblib

# =====================================================
# Load Trained Model
# =====================================================
model = joblib.load("titanic_prediction_model.pkl")

# =====================================================
# Prediction Function
# =====================================================
def predict_survival(age, pclass, fare):

    input_data = pd.DataFrame(
        {
            "age": [age],
            "pclass": [pclass],
            "fare": [fare]
        }
    )

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
        probability_text = f"\n\nSurvival Probability: {probability:.2%}"
    except:
        probability_text = ""

    if prediction == 1:
        result = "✅ Passenger is likely to SURVIVE"
    else:
        result = "❌ Passenger is NOT likely to SURVIVE"

    return result + probability_text


# =====================================================
# CSS
# =====================================================
css = """
.gradio-container{
    background:#f3f4f6;
}

.glass{
    background:white;
    padding:20px;
    border-radius:15px;
}
"""

# =====================================================
# Gradio Interface
# =====================================================
with gr.Blocks(css=css, title="Titanic Survival Prediction") as demo:

    with gr.Column(elem_classes="glass"):

        gr.Markdown(
            """
# 🚢 Titanic Survival Prediction

Predict whether a passenger is likely to survive using a trained Logistic Regression model.
"""
        )

        with gr.Row():

            with gr.Column():

                age = gr.Number(label="Age")

                pclass = gr.Dropdown(
                    choices=[1, 2, 3],
                    value=3,
                    label="Passenger Class"
                )

                fare = gr.Number(label="Fare")

                predict_btn = gr.Button("Predict Survival")

                output = gr.Textbox(label="Prediction")

            with gr.Column():

                gr.Markdown("""
## 👩‍💻 Developer

**Name:** Manya Singla

**College:** Panipat Institute of Engineering and Technology

**Model:** Logistic Regression

**Language:** Python

**Framework:** Gradio
""")

        predict_btn.click(
            fn=predict_survival,
            inputs=[age, pclass, fare],
            outputs=output
        )

# =====================================================
# Launch
# =====================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
