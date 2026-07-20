import os
import gradio as gr
import pandas as pd
import joblib

# ======================================================
# Load Trained Model
# ======================================================
model = joblib.load("titanic_prediction_model.pkl")

# ======================================================
# Prediction Function
# ======================================================
def predict_survival(age, pclass, fare):

    input_data = pd.DataFrame(
        [[age, pclass, fare]],
        columns=["age", "pclass", "fare"]
    )

    prediction = model.predict(input_data)[0]

    # Check if model supports probability prediction
    try:
        probability = model.predict_proba(input_data)[0][1]
        prob_text = f"\n\nSurvival Probability: {probability:.2%}"
    except:
        prob_text = ""

    if prediction == 1:
        result = "✅ Passenger is likely to SURVIVE"
    else:
        result = "❌ Passenger is NOT likely to survive"

    return result + prob_text


# ======================================================
# CSS
# ======================================================
custom_css = """
.gradio-container{
    background-image:url('https://images.unsplash.com/photo-1518546305927-5a555bb7020d');
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

.glass{
    background:rgba(255,255,255,0.95)!important;
    border-radius:15px;
    padding:20px;
}

.gr-button{
    background:#2563eb!important;
    color:white!important;
}

h1,h2,h3,p{
    color:#111827!important;
}
"""

# ======================================================
# Gradio Interface
# ======================================================
with gr.Blocks(css=custom_css, title="Titanic Survival Prediction") as demo:

    with gr.Column(elem_classes="glass"):

        gr.Markdown("""
# 🚢 Titanic Survival Prediction System

Predict whether a passenger is likely to survive using a **Logistic Regression Machine Learning Model**.
""")

        with gr.Row():

            # Left Side
            with gr.Column(scale=2):

                age = gr.Number(label="Age")

                pclass = gr.Dropdown(
                    choices=[1, 2, 3],
                    value=3,
                    label="Passenger Class"
                )

                fare = gr.Number(label="Fare")

                predict_btn = gr.Button(
                    "Predict Survival",
                    variant="primary"
                )

                output = gr.Textbox(label="Prediction Result")

            # Right Side
            with gr.Column(scale=1):

                gr.Markdown("""
## 👩‍💻 Developer Details

**Name:** Manya Singla

**College:** Panipat Institute of Engineering and Technology

**Project:** Titanic Survival Prediction System

**Machine Learning Model:** Logistic Regression

### 🛠 Technologies Used

- Python
- Pandas
- Scikit-Learn
- Joblib
- Gradio

📧 **Email:** manyasingla25@gmail.com

📸 **Instagram:** @manya_singla_25
""")

        predict_btn.click(
            fn=predict_survival,
            inputs=[
                age,
                pclass,
                fare
            ],
            outputs=output
        )

# ======================================================
# Launch App
# ======================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
