import pandas as pd
import streamlit as st

from titanic_analysis import evaluate

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("Titanic Survival Predictor")
st.write("Compare five classification models and estimate survival probability for a passenger profile.")

@st.cache_data
def fitted_results():
    return evaluate()

metrics, fitted = fitted_results()
st.subheader("Model comparison")
st.dataframe(metrics.style.format({column: "{:.3f}" for column in metrics.columns[1:]}), use_container_width=True, hide_index=True)

st.subheader("Passenger prediction")
left, right = st.columns(2)
with left:
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["female", "male"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sib_sp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0)
with right:
    parch = st.number_input("Parents/children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=25.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    model_name = st.selectbox("Model", list(fitted.keys()), index=0)

passenger = pd.DataFrame([{"Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sib_sp, "Parch": parch, "Fare": fare, "Embarked": embarked}])
probability = fitted[model_name].predict_proba(passenger)[0, 1]
with st.container(border=True):
    st.metric("Estimated survival probability", f"{probability:.1%}")
    st.write("Prediction:", "Likely survived" if probability >= 0.5 else "Likely did not survive")
