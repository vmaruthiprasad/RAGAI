import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
# Configure the Streamlit page title, icon, and wide layout mode
st.set_page_config(
    page_title="Advanced Diabetes Dashboard", page_icon="🩺", layout="wide"
)

# Render main title and descriptive header text
st.title("🩺 Advanced Diabetes Dashboard (Pima Indians Dataset)")
st.markdown(
    "Explore comprehensive patient demographics, clinical markers, and correlations using interactive filters and multi-variable charts."
)


# ==========================================
# 2. DATA LOADING & CACHING
# ==========================================
# Load the CSV file into a pandas DataFrame.
# Using @st.cache_data ensures the dataset is only loaded once, optimizing performance.
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df


df = load_data()


# ==========================================
# 3. SIDEBAR CONTROLS & FILTERING LOGIC
# ==========================================
st.sidebar.header("Filter Options")

# Filter 1: Age Range Slider
min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider(
    "Filter by Age Range", min_age, max_age, (min_age, max_age)
)

# Filter 2: Glucose Range Slider
min_gluc, max_gluc = int(df["Glucose"].min()), int(df["Glucose"].max())
glucose_range = st.sidebar.slider(
    "Filter by Glucose Range", min_gluc, max_gluc, (min_gluc, max_gluc)
)

# Filter 3: Outcome Selection Dropdown
outcome_filter = st.sidebar.selectbox(
    "Filter by Outcome", ["All", "Diabetic (1)", "Non-Diabetic (0)"]
)

# Apply numerical range filters to the dataset
filtered_df = df[
    (df["Age"] >= age_range[0])
    & (df["Age"] <= age_range[1])
    & (df["Glucose"] >= glucose_range[0])
    & (df["Glucose"] <= glucose_range[1])
]

# Apply categorical outcome filter based on sidebar selection
if outcome_filter == "Diabetic (1)":
    filtered_df = filtered_df[filtered_df["Outcome"] == 1]
elif outcome_filter == "Non-Diabetic (0)":
    filtered_df = filtered_df[filtered_df["Outcome"] == 0]


# ==========================================
# 4. KEY METRICS DISPLAY SECTION
# ==========================================
st.subheader("Key Metrics")

# Create 4 columns to display high-level summary statistics
col1, col2, col3, col4 = st.columns(4)

# Total filtered patient count
col1.metric("Total Patients", len(filtered_df))

# Prevalence percentage of diabetes in the filtered cohort
col2.metric(
    "Diabetes Prevalence",
    f"{(filtered_df['Outcome'].mean() * 100):.1f}%"
    if len(filtered_df) > 0
    else "0%",
)

# Average glucose level of the filtered cohort
col3.metric(
    "Average Glucose",
    f"{filtered_df['Glucose'].mean():.1f}"
    if len(filtered_df) > 0
    else "0.0",
)

# Average Body Mass Index (BMI) of the filtered cohort
col4.metric(
    "Average BMI",
    f"{filtered_df['BMI'].mean():.1f}" if len(filtered_df) > 0 else "0.0",
)

st.markdown("---")


# ==========================================
# 5. DATA PREVIEW TABLE
# ==========================================
st.subheader("Dataset Preview")
# Display the first 10 rows of the filtered dataset in an interactive table
st.dataframe(filtered_df.head(10), use_container_width=True)


# ==========================================
# 6. VISUALIZATION SECTION
# ==========================================
st.subheader("Visualizations")

# --- Row 1: Chart 1 & Chart 2 ---
col_chart1, col_chart2 = st.columns(2)

# Chart 1: Scatter plot showing BMI vs. Glucose, color-coded by Diabetes Outcome
with col_chart1:
    st.markdown("### Glucose vs. BMI by Outcome")
    fig, ax = plt.subplots(figsize=(7, 4))
    if not filtered_df.empty:
        sns.scatterplot(
            data=filtered_df,
            x="BMI",
            y="Glucose",
            hue="Outcome",
            palette={0: "royalblue", 1: "crimson"},
            alpha=0.8,
            ax=ax,
        )
        ax.set_xlabel("BMI")
        ax.set_ylabel("Glucose Level")
        ax.legend(title="Outcome", labels=["Non-Diabetic", "Diabetic"])
    st.pyplot(fig)

# Chart 2: Stacked histogram showing age distribution separated by outcome
with col_chart2:
    st.markdown("### Age Distribution by Outcome")
    fig, ax = plt.subplots(figsize=(7, 4))
    if not filtered_df.empty:
        sns.histplot(
            data=filtered_df,
            x="Age",
            hue="Outcome",
            multiple="stack",
            palette={0: "royalblue", 1: "crimson"},
            bins=15,
            ax=ax,
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
    st.pyplot(fig)


# --- Row 2: Chart 3 & Chart 4 ---
col_chart3, col_chart4 = st.columns(2)

# Chart 3: Heatmap displaying correlation coefficients between dataset features
with col_chart3:
    st.markdown("### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 4))
    if not filtered_df.empty:
        corr = filtered_df.corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            cbar=True,
            ax=ax,
            annot_kws={"size": 7},
        )
        ax.tick_params(axis="both", which="major", labelsize=8)
    st.pyplot(fig)

# Chart 4: Scatter plot comparing Skin Thickness and Insulin levels
with col_chart4:
    st.markdown("### Insulin vs. Skin Thickness")
    fig, ax = plt.subplots(figsize=(7, 4))
    if not filtered_df.empty:
        sns.scatterplot(
            data=filtered_df,
            x="SkinThickness",
            y="Insulin",
            hue="Outcome",
            palette={0: "royalblue", 1: "crimson"},
            alpha=0.8,
            ax=ax,
        )
        ax.set_xlabel("Skin Thickness")
        ax.set_ylabel("Insulin Level")
        ax.legend(title="Outcome", labels=["Non-Diabetic", "Diabetic"])
    st.pyplot(fig)