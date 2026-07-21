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

    input_data = pd.DataFrame({
        "age": [age],
        "pclass": [pclass],
        "fare": [fare]
    })

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
        probability_text = (
            f"<br><br><b>Survival Probability:</b> {probability:.2%}"
        )
    except Exception:
        probability_text = ""

    if prediction == 1:
        result = """
        <span style="color:green;font-size:22px;font-weight:bold;">
        ✅ Passenger is likely to SURVIVE
        </span>
        """
    else:
        result = """
        <span style="color:red;font-size:22px;font-weight:bold;">
        ❌ Passenger is NOT likely to SURVIVE
        </span>
        """

    return result + probability_text


# =====================================================
# Custom CSS
# =====================================================
css = """
body{
    background:#f3f4f6;
}

.gradio-container{
    max-width:1200px !important;
    margin:auto;
}

/* Developer Card */

.dev-card{
    background:#ffffff !important;
    color:#000000 !important;
    border:2px solid #2563eb;
    border-radius:15px;
    padding:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.dev-card *{
    color:#000000 !important;
}

.dev-card h2{
    color:#2563eb !important;
    text-align:center;
    margin-bottom:15px;
}

.dev-card p{
    color:#000000 !important;
    font-size:17px;
    margin:10px 0;
}

.dev-card b{
    color:#000000 !important;
}

/* Prediction Output */

.output-class{
    font-size:18px !important;
    font-weight:bold;
}
"""

# =====================================================
# Interface
# =====================================================
with gr.Blocks(
    css=css,
    title="Titanic Survival Prediction"
) as demo:

    gr.Markdown(
        """
# 🚢 Titanic Survival Prediction

Predict whether a passenger is likely to survive using a trained **Logistic Regression** model.
"""
    )

    with gr.Row():

        # Left Column
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

            output = gr.HTML()

        # Right Column
        with gr.Column(scale=1):

            gr.HTML("""
<div class="dev-card">

<h2>👩‍💻 Developer Details</h2>

<p><b>Name:</b> Manya Singla</p>

<p><b>College:</b> Panipat Institute of Engineering and Technology</p>

<p><b>Machine Learning Model:</b> Logistic Regression</p>

<p><b>Programming Language:</b> Python</p>

<p><b>Framework:</b> Gradio</p>

<p><b>Deployment:</b> Render</p>

</div>
""")

    predict_btn.click(
        fn=predict_survival,
        inputs=[age, pclass, fare],
        outputs=output
    )

# =====================================================
# Launch App
# =====================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
