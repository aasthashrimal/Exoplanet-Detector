import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import os
import io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.compose._column_transformer as ct

class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList

# Streamlit app to load a saved pipeline (joblib/pkl) and provide single-record and batch predictions.

st.set_page_config(page_title="Exoplanet Classifier", page_icon="🚀", layout="wide")

WORKDIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [WORKDIR / "xgb_exoplanet_pipeline.joblib", WORKDIR / "best_pipeline.pkl", WORKDIR / "best_pipeline.joblib"]

# --- GLOBAL CONSTANTS ---
# Placeholder for missing categorical data, expected by the ML pipeline
CAT_MISSING_PLACEHOLDER = "__NULL_CATEGORY__"

# Mappings for categorical features requested by the user.
# CRITICAL UPDATE: Code 0 is now reserved for missing/null data. All others are shifted.
ENCODING_MAPS = {
    'koi_disposition': {
        0: CAT_MISSING_PLACEHOLDER, # New: Code 0 represents missing data
        1: 'FALSE POSITIVE',
        2: 'CANDIDATE',
        3: 'CONFIRMED'
    },
    'koi_parm_prov': {
        0: CAT_MISSING_PLACEHOLDER, # New: Code 0 represents missing data
        1: 'Kepler',
        2: 'TESS',
        3: 'Other'
    },
    'koi_fittype': {
        0: CAT_MISSING_PLACEHOLDER, # New: Code 0 represents missing data
        1: 'Kepler-4',
        2: 'Kepler-5',
        3: 'Kepler-6',
        4: 'Other'
    },
    'koi_datalink_dvr': {
        0: CAT_MISSING_PLACEHOLDER, # New: Code 0 represents missing data
        1: 'No DV report',
        2: 'DV report exists'
    },
    'discoverymethod': {
        0: CAT_MISSING_PLACEHOLDER, # New: Code 0 represents missing data
        1: 'Transit',
        2: 'Radial Velocity',
        3: 'Other'
    }
}
# Reverse map for easy lookup in the guide
REVERSE_ENCODING_MAPS = {k: {v: i for i, v in v.items()} for k, v in ENCODING_MAPS.items()}


# --- DEFINITIVE LIST OF ALL REQUIRED FEATURES (91 columns) ---
ALL_REQUIRED_FEATURES = sorted(list({
    'koi_prad', 'pl_orbeccen', 'koi_ror', 'sy_vmag', 'physical_realism_score', 'planet_star_radius_ratio', 'pl_eqt', 
    'sy_kmag', 'is_multi_planet_system', 'koi_sma', 'pl_transit_probability', 'pl_st_radius_ratio', 'discoverymethod', 
    'koi_period_err1', 'st_pmra', 'st_pmraerr1', 'pl_escape_velocity', 'pl_trandeperr1', 'st_loggerr1', 'ttv_flag', 
    'pl_insol', 'st_logg_missing', 'koi_smass', 'pl_orbpererr1', 'koi_max_mult_ev', 'ra', 'pl_rade', 'st_lum', 
    'st_tefferr2', 'st_raderr1', 'koi_disposition', 'pl_trandep', 'st_teff_missing', 'koi_teq', 'pl_masse', 'sy_dist', 
    'pl_st_mass_ratio', 'rastr_10h06m26.93s', 'st_tmagerr1', 'pl_stellar_flux', 'st_tefferr1', 'toi', 'dec', 
    'pl_trandurh', 'st_dist', 'koi_ror_err2', 'pl_controv_flag', 'koi_slogg', 'koi_parm_prov', 'koi_num_transits', 
    'pl_pnum', 'st_logg', 'pl_orbper', 'st_rad', 'detection_quality_score', 'st_mass', 'koi_datalink_dvr', 
    'pl_hill_radius', 'st_tmag', 'st_pmdec', 'depth_precision', 'koi_impact', 'koi_max_sngle_ev', 'st_teff', 
    'koi_duration', 'transit_depth', 'koi_ror_err1', 'pl_orbincl', 'pl_dens', 'pl_orbsmax', 'is_grazing_transit', 
    'default_flag', 'koi_srad', 'st_disterr1', 'rastr_01h34m22.42s', 'koi_fittype', 'period_precision', 
    'pl_radeerr1', 'koi_kepmag', 'koi_period_err2', 'koi_depth_err1', 'disposition', 'pl_trandurherr1', 
    'koi_period', 'rastr_03h41m50.17s', 'tran_flag', 'ttv_flag', 'pl_controv_flag', 
    'decstr_-30d27m14.82s', 'planets_in_system', 'koi_srho', 'toipfx', 'koi_depth', 'koi_dor', 'koi_depth_err2'
}))
# -------------------------------------------------------------------

# --- Hardcoded sample data provided by the user for realistic input ---
RAW_SAMPLE_EXOPLANET = {
    # Numeric Features
    'toi': 123.01, 'toipfx': 123, 'pl_pnum': 1, 'ra': 289.92, 'dec': 44.50, 'st_pmra': 0.02, 
    'st_pmraerr1': 0.001, 'st_pmdec': -0.01, 'pl_orbper': 289.9, 'pl_orbpererr1': 0.1, 
    'pl_trandurh': 12.0, 'pl_trandurherr1': 0.2, 'pl_trandep': 0.0025, 'pl_trandeperr1': 0.0001, 
    'pl_rade': 2.4, 'pl_radeerr1': 0.1, 'pl_insol': 1.1, 'pl_eqt': 300, 'st_tmag': 11.7, 
    'st_tmagerr1': 0.01, 'st_dist': 200, 'st_disterr1': 5, 'st_teff': 5500, 'st_tefferr1': 50, 
    'st_tefferr2': 50, 'st_logg': 4.4, 'st_loggerr1': 0.05, 'st_rad': 1.0, 'st_raderr1': 0.05, 
    'st_logg_missing': 0, 'st_teff_missing': 0, 'koi_period': 289.9, 'koi_prad': 2.4, 
    'koi_sma': 0.85, 'koi_teq': 300, 'koi_impact': 0.3, 'koi_duration': 12.0, 'koi_depth': 0.0025, 
    'koi_slogg': 4.4, 'koi_srad': 1.0, 'koi_kepmag': 11.7, 'koi_smass': 1.0, 
    'planet_star_radius_ratio': 0.024, 'transit_depth': 0.0025, 'koi_max_sngle_ev': 12, 
    'koi_max_mult_ev': 1, 'koi_dor': 0.95, 'koi_ror': 0.024, 'koi_srho': 1.0, 'koi_period_err1': 0.1, 
    'koi_period_err2': 0.1, 'koi_depth_err1': 0.0001, 'koi_depth_err2': 0.0001, 'koi_ror_err1': 0.001, 
    'koi_ror_err2': 0.001, 'koi_num_transits': 3, 'period_precision': 1, 'depth_precision': 1, 
    'is_grazing_transit': 0, 'planets_in_system': 1, 'is_multi_planet_system': 0, 
    'physical_realism_score': 0.9, 'detection_quality_score': 0.95, 'pl_orbsmax': 0.85, 
    'pl_orbeccen': 0.02, 'pl_orbincl': 89.5, 'pl_masse': 5.0, 'pl_dens': 5.5, 'st_mass': 1.0, 
    'st_lum': 1.0, 'sy_dist': 200, 'sy_vmag': 11.7, 'sy_kmag': 10.5, 'pl_st_radius_ratio': 0.024, 
    'pl_st_mass_ratio': 0.005, 'pl_escape_velocity': 12.5, 'pl_stellar_flux': 1.1, 
    'pl_hill_radius': 0.01, 'pl_transit_probability': 0.01, 'disposition': 1, 'default_flag': 0, 
    'tran_flag': 1, 'ttv_flag': 0, 'pl_controv_flag': 0, 
    # Categorical features as their default integer codes for the input interface
    'koi_disposition': 2, # CANDIDATE 
    'koi_parm_prov': 1, # Kepler 
    'koi_fittype': 1, # Kepler-4
    'koi_datalink_dvr': 2, # DV report exists
    'discoverymethod': 1, # Transit
    # Example OHE column
    'rastr_10h06m26.93s': 1,
}

# Process sample data to ensure all 91 features are present
REALISTIC_SAMPLE_EXOPLANET = {}
for feature in ALL_REQUIRED_FEATURES:
    if feature in RAW_SAMPLE_EXOPLANET:
        REALISTIC_SAMPLE_EXOPLANET[feature] = RAW_SAMPLE_EXOPLANET[feature]
    elif feature.startswith(('rastr_', 'decstr_')):
        # OHE binary columns
        REALISTIC_SAMPLE_EXOPLANET[feature] = 0
    else:
        # Default for other numeric columns
        REALISTIC_SAMPLE_EXOPLANET[feature] = 0.0


@st.cache_resource
def load_pipeline():
    """Try to load the first existing model file. Returns (pipeline, model_path) or (None, None)."""
    for p in MODEL_CANDIDATES:
        if p.exists():
            try:
                # Use joblib.load for .joblib and .pkl files for consistency
                model = joblib.load(p)
                return model, p.name
            except Exception as e:
                st.warning(f"Failed to load {p.name}: {e}")
    return None, None


def extract_feature_info(pipeline):
    """
    Identifies numeric and raw categorical features based on the complete list.
    """
    
    all_features = ALL_REQUIRED_FEATURES
    categorical_features = list(ENCODING_MAPS.keys())
    numeric_features = [f for f in all_features if f not in categorical_features]

    return {
        "numeric_features": numeric_features, 
        "categorical_features": categorical_features, 
        "all_features": all_features
    }


def map_input_to_strings(df, categorical_features):
    """
    CRITICAL FUNCTION: Takes a DataFrame where categorical columns contain 
    integer codes (0, 1, 2...) and maps them to the string values the pipeline expects.
    
    Code 0 maps to the CAT_MISSING_PLACEHOLDER.
    """
    df_mapped = df.copy()
    
    for col in categorical_features:
        if col in ENCODING_MAPS:
            mapping = ENCODING_MAPS[col]
            
            # 1. Ensure the column is numeric (float/int) for mapping to work
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')

            # 2. Use .map() for efficient and direct code-to-string conversion.
            df_mapped[col] = df_mapped[col].map(mapping)
            
            # 3. Fill any resulting NaN (from true missing data or invalid codes) 
            # with the placeholder.
            df_mapped[col] = df_mapped[col].fillna(CAT_MISSING_PLACEHOLDER).astype(str)
        else:
            # Fallback for unmapped categoricals
            df_mapped[col] = df_mapped[col].astype(str).replace('', CAT_MISSING_PLACEHOLDER)
            
    return df_mapped


def feature_descriptions(all_features):
    """Provides a brief description for each column."""
    descs = {}
    for f in all_features:
        key = f.lower()
        
        d = ""
        if 'err' in key: d = "Error/Uncertainty for this measurement."
        elif 'toi' in key or 'koi' in key: d = "Kepler/TESS Object of Interest Identifier."
        elif any(tok in key for tok in ("rade", "radius", "rad", "prad", "srad")): d = "Radius (planetary or stellar)."
        elif any(tok in key for tok in ("period", "orbper")): d = "Orbital period (days)."
        elif any(tok in key for tok in ("depth", "trandep")): d = "Transit depth (fractional flux loss)."
        elif any(tok in key for tok in ("dur", "duration")): d = "Transit duration (hours)."
        elif any(tok in key for tok in ("teff", "temp")): d = "Stellar effective temperature ($\text{K}$)."
        elif 'logg' in key: d = "Stellar surface gravity ($\text{log g}$)."
        elif any(tok in key for tok in ("mag", "tmag", "vmag")): d = "Apparent magnitude."
        elif any(tok in key for tok in ("mass", "smass", "masse")): d = "Mass (planetary or stellar)."
        elif 'dist' in key: d = "Distance (parsecs)."
        elif 'flag' in key or 'missing' in key or 'is_' in key: d = "Binary flag (0 or 1)."
        elif any(tok in key for tok in ("rastr", "decstr")): d = "Internal one-hot encoded coordinate flag (0 or 1)."
        else: d = "General measurement or derived property."

        descs[f] = d
    return descs


def plot_probabilities(prob_false, prob_planet):
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["False Positive", "Planet Candidate"]
    probs = [prob_false, prob_planet]
    colors = ["#ff6b6b", "#51cf66"]
    ax.bar(labels, probs, color=colors)
    ax.set_ylim(0, 1)
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("Probability")
    plt.tight_layout()
    return fig


def main():
    st.title("🚀 Exoplanet Classifier — Streamlit")

    pipeline, model_name = load_pipeline()
    if pipeline is None:
        st.error("No model file found. Place 'xgb_exoplanet_pipeline.joblib' or 'best_pipeline.pkl' in this folder.")
        return

    st.sidebar.header("Model")
    st.sidebar.success(f"Loaded: {model_name}")

    feature_info = extract_feature_info(pipeline)
    all_features = feature_info.get("all_features", [])
    numeric_features = feature_info.get("numeric_features", [])
    categorical_features = feature_info.get("categorical_features", [])
    
    # --- Sidebar Controls ---
    with st.sidebar.expander("🔧 Controls", expanded=True):
        if st.button("Load sample input", key="load_sample_sidebar"):
            st.session_state["sample_loaded"] = True
        st.write("---")

    # Provide a sample CSV users can download and edit to match model expectations
    if all_features:
        sample_df = pd.DataFrame([REALISTIC_SAMPLE_EXOPLANET], columns=all_features)
        
        sample_csv = sample_df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("Download sample CSV (with codes)", sample_csv, file_name="sample_input_with_codes.csv", mime="text/csv")
        st.sidebar.write(f"Model requires **{len(all_features)}** columns.")
    st.sidebar.markdown("---")

    # --- Main Content: Tabs ---
    tab1, tab2 = st.tabs(["Prediction", "CSV Format Guide"])

    with tab1:
        st.header("Batch prediction")
        uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="main_batch_upload")
        if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    st.write(f"Rows in file: {len(df)}")
                    
                    expected = all_features
                    
                    # 1. Validation and filling for batch
                    missing = [c for c in expected if c not in df.columns]
                    if missing:
                        st.warning(f"Missing columns detected in upload; filling with defaults (0 for missing code or 0.0 for numeric). (first 5 shown): {missing[:5]}")
                        for c in missing:
                            if c in numeric_features:
                                df[c] = 0.0
                            elif c in categorical_features:
                                # For missing categorical column, use the code 0 (which now maps to __NULL_CATEGORY__)
                                df[c] = 0 
                            else:
                                df[c] = 0.0

                    # 2. Ensure all columns are present and in the correct order
                    df = df[expected]

                    # 3. Explicitly cast numeric columns
                    for c in numeric_features:
                        # Ensure all numeric/OHE columns are float, filling NaNs with 0.0
                        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
                        
                    # 4. Map the input codes/values back to the model's expected strings
                    df_mapped = map_input_to_strings(df, categorical_features)
                    
                    # 5. Prediction
                    preds = pipeline.predict(df_mapped)
                    probs = None
                    if hasattr(pipeline, "predict_proba"):
                        probs = pipeline.predict_proba(df_mapped)
                    
                    # 6. Results
                    results = df.copy() # Use the original DF with codes for display
                    results["prediction"] = preds
                    has_probs = False
                    if probs is not None and getattr(probs, 'shape', None) and len(probs.shape) > 1 and probs.shape[1] == 2:
                        results["prob_false_positive"] = probs[:, 0]
                        results["prob_planet_candidate"] = probs[:, 1]
                        has_probs = True

                    # Build a compact preview
                    id_candidates = [c for c in df.columns if c.lower() in ("toi", "koi", "kepoi", "id", "object_id")]
                    preview_cols = []
                    if id_candidates:
                        preview_cols.append(id_candidates[0])
                    preview_cols.append("prediction")
                    if has_probs:
                        preview_cols.extend(["prob_false_positive", "prob_planet_candidate"])
                    
                    preview_cols = [c for c in preview_cols if c in results.columns]
                    preview_df = results[preview_cols].head(200)

                    st.subheader("Batch prediction results (preview)")
                    st.dataframe(preview_df)

                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button("Download full results CSV", csv, file_name="predictions_full.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Failed batch prediction: {e}")

        st.markdown("---")

        st.header("Single-record prediction")

        # Single-record prediction logic
        if not all_features:
            st.warning("Could not infer feature list from pipeline.")
        
        # Build defaults for every expected feature using the complete, filtered sample
        defaults = {c: REALISTIC_SAMPLE_EXOPLANET.get(c, 0) for c in all_features}
        
        # State management for loading sample data
        if st.session_state.get("sample_loaded"):
            for k, v in REALISTIC_SAMPLE_EXOPLANET.items():
                if k in defaults:
                    st.session_state[f"input_{k}"] = v
            st.session_state["sample_loaded"] = False


        with st.form("single_input"):
            cols = st.columns(3)
            input_row = {} # Initialize empty dict to store form values

            # Input generation loop
            feature_list = all_features
            descs = feature_descriptions(all_features)
            
            for i, feat in enumerate(feature_list):
                col = cols[i % 3]
                default_value = defaults.get(feat)
                state_key = f"input_{feat}"
                
                # Custom-encoded categorical features use a selectbox
                if feat in ENCODING_MAPS:
                    options_dict = ENCODING_MAPS[feat]
                    # Filter out the CAT_MISSING_PLACEHOLDER for the user-friendly dropdown
                    # Only show codes 1 and above for valid data
                    display_options_dict = {k: v for k, v in options_dict.items() if v != CAT_MISSING_PLACEHOLDER}
                    
                    # --- CHANGE HERE: Simplified Display Strings ---
                    # Add the "Missing" option as the first item for the dropdown (corresponds to code 0)
                    display_options = [f"0 -> MISSING (Leave Empty)"]
                    
                    # Add all other valid options
                    for code, desc in display_options_dict.items():
                         display_options.append(f"{code} -> {desc}")
                    
                    # Determine default selection index for the sample data
                    default_index = 0
                    if default_value in options_dict and options_dict[default_value] != CAT_MISSING_PLACEHOLDER:
                         # Find the index in the *display_options* list (offset by 1 because of 'Missing' at index 0)
                         value_str = f"{default_value} -> {options_dict[default_value]}"
                         try:
                             default_index = display_options.index(value_str)
                         except ValueError:
                             default_index = 0 # Fallback to Missing if something is wrong

                    input_value = col.selectbox(
                        feat, 
                        options=display_options, 
                        index=default_index,
                        help=f"**Categorical Code:** {descs.get(feat, '')}"
                    )
                    # Extract the integer code from the selection string (e.g., '2 -> CANDIDATE' -> 2)
                    input_row[feat] = int(input_value.split(' ')[0])
                else:
                    # Numeric and OHE binary inputs
                    input_row[feat] = col.number_input(
                        feat, 
                        value=float(default_value if default_value is not None else 0.0), 
                        format="%.6f",
                        key=state_key,
                        help=descs.get(feat, '')
                    )


            submitted = st.form_submit_button("Analyze object")

            if submitted:
                try:
                    # 1. Create the DataFrame using the dictionary from the form
                    input_df = pd.DataFrame([input_row])

                    # 2. Ensure column order matches the model's requirement
                    input_df = input_df[all_features]

                    # 3. Explicitly cast numeric columns
                    for c in numeric_features:
                        input_df[c] = pd.to_numeric(input_df[c], errors='coerce').fillna(0.0)
                        
                    # CRITICAL STEP for single record: Ensure categorical input is treated as an integer
                    for c in categorical_features:
                        input_df[c] = input_df[c].astype(int)
                    
                    # Convert the input codes back to the required strings using the robust mapper
                    input_df_mapped = map_input_to_strings(input_df, categorical_features)
                    
                    # 4. Perform prediction
                    pred = pipeline.predict(input_df_mapped)[0]
                    probs = None
                    if hasattr(pipeline, "predict_proba"):
                        probs = pipeline.predict_proba(input_df_mapped)[0]

                    if probs is not None and len(probs) == 2:
                        prob_false, prob_planet = float(probs[0]), float(probs[1])
                    else:
                        prob_false, prob_planet = (1.0, 0.0) if pred == 0 else (0.0, 1.0)

                    label = "Planet Candidate" if pred == 1 else "False Positive"
                    st.metric("Prediction", label)
                    st.metric("Confidence", f"{max(prob_false, prob_planet):.2%}")

                    st.subheader("Probabilities")
                    fig = plot_probabilities(prob_false, prob_planet)
                    st.pyplot(fig)

                    st.subheader("Input used for prediction (Mapped to Model Strings)")
                    st.dataframe(input_df_mapped.T)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

    with tab2:
        st.header("Exoplanet Classification CSV Format Guide (91 Columns)")
        st.markdown(
            """
            This table lists all **91 required columns** for batch prediction. Your CSV must contain these columns
            and use the precise codes for the categorical features.

            ---
            ### ⚠️ Special Rule for Categorical Columns
            For the 5 categorical columns listed below, you may **leave the cell empty for missing data**.
            
            ---
            ### Categorical Feature Codes (Valid Data Only)
            """
        )
        
        # Dynamic table generation for categorical codes (excluding 0/missing)
        cat_table_markdown = "| Column Name | Required Code | Definition |\n| :--- | :--- | :--- |\n"
        
        for col, mapping in ENCODING_MAPS.items():
            is_first = True
            # Iterate through the codes and definitions
            for code, definition in mapping.items():
                if definition == CAT_MISSING_PLACEHOLDER:
                    # Skip the missing data row as requested
                    continue
                
                # Determine the text for the first column
                col_name_display = f"**{col}**" if is_first else ""
                
                cat_table_markdown += f"| {col_name_display} | {code} | {definition} |\n"
                is_first = False

        st.markdown(cat_table_markdown)
        
        st.markdown(
            """
            ---
            ### Full List of Required Columns
            If you do not have data for a numeric column, use **$\\text{0.0}$**. For categorical columns, use the appropriate integer code (or leave the cell empty for missing data).
            """
        )
        
        # Generate the full list of 91 columns
        descs = feature_descriptions(all_features)
        
        # Prepare data for the full DataFrame display
        guide_data = []
        for i, col in enumerate(all_features):
            is_cat = col in ENCODING_MAPS
            guide_data.append({
                '#': i + 1,
                'Column Name': f"**{col}** {'(CATEGORICAL)' if is_cat else ''}",
                'Description': descs.get(col, 'General measurement or derived property.'),
            })
            
        guide_df = pd.DataFrame(guide_data)
        st.dataframe(guide_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

