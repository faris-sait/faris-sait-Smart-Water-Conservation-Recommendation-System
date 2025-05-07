import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="WaterWise Pro - Smart Conservation System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load custom CSS
def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{css_file}' not found. Using default styling.")


# Load enhanced CSS
load_css('style.css')


# App state management
class AppState:
    def __init__(self):
        self.prediction = None
        self.recommendations = []
        self.savings = 0
        self.historical_data = None


# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState()
    st.session_state.show_advanced = False
    st.session_state.history = []
    st.session_state.first_run = True


# Data loading with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('water_usage_data.csv')
        # Data cleaning
        df = df.dropna()  # Remove missing values
        # Handle outliers - cap extreme values
        for col in ['WaterUsage_Liters', 'HouseholdSize', 'OutdoorUsage_Percent']:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q3)
        return df
    except FileNotFoundError:
        # Use sample data if file not found
        st.warning("Water usage data file not found. Using sample data.")
        return generate_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return generate_sample_data()


# Generate sample data for demo purposes or when file is missing
def generate_sample_data(n_samples=1000):
    activities = ["Showering", "Washing Clothes", "Gardening", "Cooking", "Other"]
    efficiencies = ["Low", "Medium", "High"]

    data = {
        'WaterUsage_Liters': np.random.gamma(shape=5, scale=10, size=n_samples),
        'HouseholdSize': np.random.randint(1, 8, size=n_samples),
        'OutdoorUsage_Percent': np.random.randint(0, 80, size=n_samples),
        'Activity': np.random.choice(activities, size=n_samples),
        'ApplianceEfficiency': np.random.choice(efficiencies, size=n_samples)
    }

    # Create usage categories based on water usage
    usage_cat = []
    for usage in data['WaterUsage_Liters']:
        if usage < 40:
            usage_cat.append('Low')
        elif usage < 100:
            usage_cat.append('Medium')
        else:
            usage_cat.append('High')

    data['UsageCategory'] = usage_cat
    return pd.DataFrame(data)


# Enhanced model training with validation metrics
@st.cache_resource
def get_model():
    data = load_data()

    # Feature engineering
    X = data[['WaterUsage_Liters', 'HouseholdSize', 'OutdoorUsage_Percent']]
    y = data['UsageCategory']

    # Encode categorical features
    le_activity = LabelEncoder()
    le_efficiency = LabelEncoder()

    X = X.copy()
    if 'Activity' in data.columns:
        X['Activity'] = le_activity.fit_transform(data['Activity'])
    if 'ApplianceEfficiency' in data.columns:
        X['ApplianceEfficiency'] = le_efficiency.fit_transform(data['ApplianceEfficiency'])

    # Feature scaling for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model with improved parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all available processors
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    model_metrics = {
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    return model, le_activity, le_efficiency, scaler, model_metrics


# Advanced recommendation engine
def get_recommendations(usage, household_size, outdoor_usage, appliance_efficiency, activity, prediction):
    recommendations = []
    savings = 0

    # Base recommendations based on usage category
    if prediction == 'High':
        recommendations.append({
            "title": "High Water Usage Detected",
            "description": "Your water usage is significantly higher than average. Consider a comprehensive water audit.",
            "impact": "High Impact",
            "savings": 30
        })
        savings += 30

        if appliance_efficiency == 'Low':
            recommendations.append({
                "title": "Upgrade Water Appliances",
                "description": "Invest in high-efficiency WaterSense-labeled products for significant long-term savings.",
                "impact": "High Impact",
                "savings": 20
            })
            savings += 20

    elif prediction == 'Medium':
        recommendations.append({
            "title": "Moderate Usage",
            "description": "Your usage is moderate. Small changes can still lead to significant savings over time.",
            "impact": "Medium Impact",
            "savings": 15
        })
        savings += 15

    else:  # Low
        recommendations.append({
            "title": "Efficient Water Usage",
            "description": "Great job! Your usage is already low. Maintain with regular monitoring.",
            "impact": "Low Impact",
            "savings": 5
        })
        savings += 5

    # Activity-specific recommendations
    if activity == 'Showering' and usage > 60:
        recommendations.append({
            "title": "Shower Optimization",
            "description": "Install a low-flow showerhead (< 2.0 GPM) and limit showers to 5 minutes using a timer.",
            "impact": "Medium Impact",
            "savings": 10
        })
        savings += 10

    elif activity == 'Washing Clothes' and usage > 80:
        recommendations.append({
            "title": "Laundry Efficiency",
            "description": "Use a high-efficiency washing machine and wash full loads only. Consider cold water washes.",
            "impact": "Medium Impact",
            "savings": 15
        })
        savings += 15

    elif activity == 'Gardening' and outdoor_usage > 30:
        recommendations.append({
            "title": "Smart Gardening",
            "description": "Switch to drip irrigation, use smart controllers, and add mulch to retain soil moisture.",
            "impact": "High Impact",
            "savings": 20
        })
        savings += 20

    elif activity == 'Cooking' and usage > 40:
        recommendations.append({
            "title": "Kitchen Water Savings",
            "description": "Install aerators on kitchen faucets and reuse cooking water for plants.",
            "impact": "Medium Impact",
            "savings": 8
        })
        savings += 8

    # Household size recommendations
    if household_size > 4 and usage > 200:
        recommendations.append({
            "title": "Family Water Plan",
            "description": "Implement a household water-saving plan with daily usage targets and rewards.",
            "impact": "High Impact",
            "savings": 10
        })
        savings += 10

    # Seasonal recommendation
    current_month = datetime.now().month
    if 5 <= current_month <= 9:  # Summer months
        recommendations.append({
            "title": "Seasonal Adjustment",
            "description": "During summer, water gardens early morning or evening to reduce evaporation loss.",
            "impact": "Medium Impact",
            "savings": 8
        })
        savings += 8

    return recommendations, savings


# Visualization functions
def create_usage_chart(data):
    fig = px.pie(
        names=data['UsageCategory'].value_counts().index,
        values=data['UsageCategory'].value_counts().values,
        title="Distribution of Water Usage Categories",
        color_discrete_sequence=px.colors.sequential.Blues,
        hole=0.4
    )
    fig.update_layout(
        legend_title="Usage Category",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def create_feature_importance_chart(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[feature_names[i] for i in indices],
            x=[importances[i] for i in indices],
            orientation='h',
            marker=dict(color='rgba(58, 71, 80, 0.6)', line=dict(color='rgba(58, 71, 80, 1.0)', width=1))
        )
    )
    fig.update_layout(
        title="Feature Importance for Water Usage Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400
    )
    return fig


def create_usage_by_activity_chart(data):
    activity_usage = data.groupby('Activity')['WaterUsage_Liters'].mean().reset_index()
    fig = px.bar(
        activity_usage,
        x='Activity',
        y='WaterUsage_Liters',
        color='WaterUsage_Liters',
        color_continuous_scale=px.colors.sequential.Blues,
        title="Average Water Usage by Activity"
    )
    fig.update_layout(xaxis_title="Activity", yaxis_title="Average Water Usage (Liters)")
    return fig


# Save prediction history
def save_prediction(usage, household_size, outdoor_usage, appliance_efficiency, activity, prediction, savings):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "timestamp": timestamp,
        "usage": usage,
        "household_size": household_size,
        "outdoor_usage": outdoor_usage,
        "appliance_efficiency": appliance_efficiency,
        "activity": activity,
        "prediction": prediction,
        "savings_potential": savings
    })


# Main App
def main():
    # Load data and models
    data = load_data()
    model, le_activity, le_efficiency, scaler, model_metrics = get_model()

    # App container with custom styling
    st.markdown('<div class="main">', unsafe_allow_html=True)

    # App Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913465.png", width=100)
    with col2:
        st.title("WaterWise Pro")
        st.markdown('<p class="subtitle">Smart Water Conservation Recommendation System</p>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.header("About WaterWise Pro")
        st.markdown("""
        <div class="sidebar-content">
            <p>This advanced system uses machine learning to analyze your water usage patterns and provide personalized recommendations for conservation.</p>
            <p>Our AI model is trained on thousands of household water usage patterns to deliver accurate predictions and meaningful savings.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # Model performance metrics in sidebar
        st.subheader("Model Performance")
        st.markdown(f"<div class='metric-card'>Accuracy: {model_metrics['accuracy']:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>Cross-validation: {np.mean(model_metrics['cv_scores']):.2f}</div>",
                    unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # Advanced options toggle
        st.session_state.show_advanced = st.checkbox("Show Advanced Options", value=st.session_state.show_advanced)

        if st.session_state.show_advanced:
            st.subheader("Advanced Options")

            # Download sample data option
            if st.button("Download Sample Data Template"):
                sample = generate_sample_data(10)
                csv = sample.to_csv(index=False)
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="water_usage_sample.csv",
                    mime="text/csv"
                )

            # View prediction history
            if st.button("View Prediction History"):
                if st.session_state.history:
                    history_df = pd.DataFrame(st.session_state.history)
                    st.dataframe(history_df)
                else:
                    st.info("No prediction history available yet.")

    # Main content area
    st.subheader("Enter Your Water Usage Details")
    st.markdown(
        '<p class="info-text">Fill out the form below to get personalized water conservation recommendations.</p>',
        unsafe_allow_html=True)

    # Enhanced input form with better layout
    with st.form("input_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)  # Using 3 columns for better space utilization

        with col1:
            usage = st.number_input(
                "Daily Water Usage (liters)",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                help="Average daily water consumption in liters"
            )

            household_size = st.slider(
                "Household Size",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of people in your household"
            )

        with col2:
            outdoor_usage = st.slider(
                "Outdoor Usage (%)",
                min_value=0,
                max_value=100,
                value=20,
                help="Percentage of water used outdoors (garden, pool, etc.)"
            )

            appliance_efficiency = st.selectbox(
                "Appliance Efficiency",
                options=["Low", "Medium", "High"],
                index=1,
                help="Overall efficiency of your water-using appliances"
            )

        with col3:
            activity = st.selectbox(
                "Primary Water Activity",
                options=["Showering", "Washing Clothes", "Gardening", "Cooking", "Other"],
                help="Activity that uses most of your water"
            )

            if st.session_state.show_advanced:
                region = st.selectbox(
                    "Climate Region",
                    options=["Arid", "Temperate", "Tropical", "Continental"],
                    index=1,
                    help="Your general climate region"
                )

        # Centered submit button with improved styling
        submitted = st.form_submit_button("Get Personalized Recommendations")

    # Process form submission
    if submitted:
        # Display loading animation
        with st.spinner("Analyzing your water usage patterns..."):
            time.sleep(0.8)  # Simulate processing time

            # Prepare input data for model
            input_features = {
                'WaterUsage_Liters': [usage],
                'HouseholdSize': [household_size],
                'OutdoorUsage_Percent': [outdoor_usage]
            }

            # Add encoded categorical features
            if 'Activity' in data.columns:
                try:
                    input_features['Activity'] = [le_activity.transform([activity])[0]]
                except ValueError:
                    # Handle unseen categories
                    input_features['Activity'] = [0]

            if 'ApplianceEfficiency' in data.columns:
                try:
                    input_features['ApplianceEfficiency'] = [le_efficiency.transform([appliance_efficiency])[0]]
                except ValueError:
                    # Handle unseen categories
                    input_features['ApplianceEfficiency'] = [1]

            # Create DataFrame and scale features
            input_df = pd.DataFrame(input_features)
            input_scaled = scaler.transform(input_df)

            # Make prediction
            prediction = model.predict(input_scaled)[0]

            # Get detailed recommendations
            recommendations, savings = get_recommendations(
                usage, household_size, outdoor_usage, appliance_efficiency, activity, prediction
            )

            # Save to history
            save_prediction(usage, household_size, outdoor_usage, appliance_efficiency, activity, prediction, savings)

            # Update app state
            st.session_state.app_state.prediction = prediction
            st.session_state.app_state.recommendations = recommendations
            st.session_state.app_state.savings = savings

    # Display results if prediction exists
    if st.session_state.app_state.prediction:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.subheader("Your Water Usage Analysis")

        # Show results in cards
        cols = st.columns(3)
        with cols[0]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Usage Category</div>
                    <div class="metric-value {st.session_state.app_state.prediction.lower()}-usage">{st.session_state.app_state.prediction}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with cols[1]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Savings Potential</div>
                    <div class="metric-value">{st.session_state.app_state.savings}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with cols[2]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Daily Usage</div>
                    <div class="metric-value">{usage:.1f} liters</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Display personalized recommendations
        st.subheader("Personalized Water Conservation Plan")

        for i, rec in enumerate(st.session_state.app_state.recommendations):
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <div class="rec-header">
                        <span class="rec-title">{rec['title']}</span>
                        <span class="rec-impact">{rec['impact']}</span>
                    </div>
                    <div class="rec-body">
                        <p>{rec['description']}</p>
                        <div class="rec-savings">Potential Savings: <b>{rec['savings']}%</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Usage comparison visualization
        st.subheader("How Your Usage Compares")

        # Get average usage for comparison
        avg_usage = data['WaterUsage_Liters'].mean()
        avg_usage_same_household = data[data['HouseholdSize'] == household_size]['WaterUsage_Liters'].mean()

        comparison_data = {
            'Category': ['Your Usage', 'Average Usage', f'Average for {household_size}-person household'],
            'Liters': [usage, avg_usage, avg_usage_same_household]
        }
        comparison_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comparison_df,
            x='Category',
            y='Liters',
            color='Category',
            color_discrete_sequence=['#4fc3f7', '#0288d1', '#01579b'],
            text='Liters',
            title="Usage Comparison"
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Water Usage (Liters)",
            showlegend=False
        )
        fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Data exploration section
    if st.checkbox("Explore Water Usage Data"):
        st.subheader("Data Insights")

        tab1, tab2, tab3 = st.tabs(["Usage Distribution", "Activity Analysis", "Feature Importance"])

        with tab1:
            st.plotly_chart(create_usage_chart(data), use_container_width=True)

            # Distribution of water usage
            fig = px.histogram(
                data,
                x="WaterUsage_Liters",
                color="UsageCategory",
                marginal="box",
                title="Distribution of Water Usage",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.plotly_chart(create_usage_by_activity_chart(data), use_container_width=True)

            # Household size vs water usage by efficiency
            fig = px.scatter(
                data,
                x="HouseholdSize",
                y="WaterUsage_Liters",
                color="ApplianceEfficiency",
                size="OutdoorUsage_Percent",
                hover_data=["Activity"],
                title="Household Size vs. Water Usage by Appliance Efficiency"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            feature_names = list(data.columns)
            feature_names.remove('UsageCategory')  # Remove target variable
            st.plotly_chart(create_feature_importance_chart(model, feature_names), use_container_width=True)

            # Display model performance details
            with st.expander("Model Performance Details"):
                st.json(model_metrics['classification_report'])

                # Confusion matrix as heatmap
                classes = list(set(data['UsageCategory']))
                cm = model_metrics['confusion_matrix']

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.title('Confusion Matrix')
                st.pyplot(fig)

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>WaterWise Pro © 2025 | Made with 💧 for a sustainable future</p>
            <p class="small-text">Data updated monthly from global water conservation agencies.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()


