import os
import gradio as gr
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ======================================================
# Load Dataset
# ======================================================
df = pd.read_csv("Titanic.csv")

# ======================================================
# Data Preprocessing
# ======================================================
df = df.dropna(subset=["age", "fare", "pclass", "survived"])

df_filtered = df[
    (df["age"] <= 48) &
    (df["age"] > 7) &
    (df["fare"] <= 27)
]

X = df_filtered[["age", "pclass", "fare"]]
y = df_filtered["survived"]

# ======================================================
# Train Model
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ======================================================
# Prediction Function
# ======================================================
def predict_survival(age, pclass, fare):

    input_data = pd.DataFrame(
        [[age, pclass, fare]],
        columns=[
            "age",
            "pclass",
            "fare"
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "✅ Passenger is likely to SURVIVE"
    else:
        result = "❌ Passenger is NOT likely to survive"

    return f"""{result}

Survival Probability : {probability:.2%}
"""

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
# Interface
# ======================================================
with gr.Blocks(
    css=custom_css,
    title="Titanic Survival Prediction"
) as demo:

    with gr.Column(elem_classes="glass"):

        gr.Markdown("""
# 🚢 Titanic Survival Prediction System

Predict whether a passenger is likely to survive using a **Logistic Regression Machine Learning Model**.
""")

        with gr.Row():

            # ================= Left ===================
            with gr.Column(scale=2):

                age = gr.Number(label="Age")

                pclass = gr.Dropdown(
                    choices=[1,2,3],
                    label="Passenger Class",
                    value=3
                )

                fare = gr.Number(label="Fare")

                predict = gr.Button(
                    "Predict Survival",
                    variant="primary"
                )

                output = gr.Textbox(
                    label="Prediction Result"
                )

            # ================= Right ==================
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
- Gradio

📧 **Email:** manyasingla25@gmail.com

📸 **Instagram:** @manya_singla_25
""")

        predict.click(
            fn=predict_survival,
            inputs=[
                age,
                pclass,
                fare
            ],
            outputs=output
        )

# ======================================================
# Launch
# ======================================================
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT",7860))
    )
