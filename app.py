import streamlit as st
import pandas as pd
import joblib
from preprocess import extract_features

model_lama = joblib.load('nyeri_rf_baseline_model.pkl')
joblib.dump(model_lama, 'nyeri_rf_baseline_model.pkl', compress=3)
