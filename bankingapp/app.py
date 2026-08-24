
import streamlit as st
import tensorflow as tf
import pickle
import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import lime
import lime.lime_text
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# ============= CONFIGURATION =============
MODEL_DIR = 'model/'
TOKENIZER_PATH = "banking_tokenizer.pkl"

# ============= LIME EXPLAINER CLASS =============

class StreamlitBankingExplainer:
    """LIME Explainer integrated for Streamlit"""

    def __init__(self, model, tokenizer, label_encoders, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoders = label_encoders
        self.max_length = max_length

        # Initialize LIME
        self.lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=None,
            split_expression=r'\W+',
            bow=False,
            random_state=42
        )

    def predict_probabilities(self, texts):
        """Prediction function for LIME"""
        processed_texts = []

        for text in texts:
            text = str(text).lower()
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
            processed_texts.append(padded[0])

        processed_array = np.array(processed_texts)
        predictions = self.model.predict(processed_array, verbose=0)
        return predictions

    def explain_prediction(self, text, head='category', num_features=8):
        """Generate LIME explanation for specific head"""

        head_info = {
            'category': {'index': 0, 'encoder': 'category_encoder'},
            'intent': {'index': 1, 'encoder': 'intent_encoder'},
            'urgency': {'index': 2, 'encoder': 'urgency_encoder'}
        }

        encoder = self.label_encoders[head_info[head]['encoder']]
        self.lime_explainer.class_names = list(encoder.classes_)

        def predict_fn(texts):
            all_preds = self.predict_probabilities(texts)
            head_preds = all_preds[head_info[head]['index']]
            return head_preds

        explanation = self.lime_explainer.explain_instance(
            text, predict_fn, num_features=num_features, num_samples=500
        )

        return explanation

# ============= LOAD MODEL AND COMPONENTS =============
@st.cache_resource
def load_model_components():
    """Load all model components with caching for better performance"""

    # Load the trained model
    model = tf.keras.models.load_model(f'{MODEL_DIR}hierarchical_3headed_banking_model_complete_V3.keras')

    # Load tokenizer
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load label encoders
    with open(f"{MODEL_DIR}label_encoders_3headed_V3.pkl", 'rb') as f:
        label_encoders = pickle.load(f)

    # Load model info
    with open(f"{MODEL_DIR}model_info_3headed_V3.json", 'r') as f:
        model_info = json.load(f)

    # Initialize explainer
    explainer = StreamlitBankingExplainer(model, tokenizer, label_encoders, model_info['max_length'])

    return model, tokenizer, label_encoders, model_info, explainer

# ============= TEXT PREPROCESSING FUNCTIONS =============

def clean_text(text):
    """Clean and preprocess text (same as training)"""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = ' '.join(text.split())

    return text

def preprocess_input_text(text, tokenizer, max_length):
    """Preprocess single input text for prediction"""
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

# ============= PREDICTION FUNCTIONS =============

def make_prediction_with_explanations(text, model, tokenizer, label_encoders, max_length, explainer):
    """Make prediction with LIME explanations"""

    try:
        # Regular prediction
        processed_text = preprocess_input_text(text, tokenizer, max_length)
        predictions = model.predict(processed_text, verbose=0)

        # Extract predictions for each head
        category_probs = predictions[0][0]
        intent_probs = predictions[1][0]
        urgency_probs = predictions[2][0]

        # Get predicted classes and confidences
        category_pred = np.argmax(category_probs)
        intent_pred = np.argmax(intent_probs)
        urgency_pred = np.argmax(urgency_probs)

        category_confidence = category_probs[category_pred]
        intent_confidence = intent_probs[intent_pred]
        urgency_confidence = urgency_probs[urgency_pred]

        # Convert to readable labels
        category_label = label_encoders['category_encoder'].inverse_transform([category_pred])[0]
        intent_label = label_encoders['intent_encoder'].inverse_transform([intent_pred])[0]
        urgency_label = label_encoders['urgency_encoder'].inverse_transform([urgency_pred])[0]

        # Generate LIME explanations
        with st.spinner('Generating explanations...'):
            category_explanation = explainer.explain_prediction(text, head='category', num_features=8)
            intent_explanation = explainer.explain_prediction(text, head='intent', num_features=8)
            urgency_explanation = explainer.explain_prediction(text, head='urgency', num_features=8)

        return {
            'category': {'label': category_label, 'confidence': float(category_confidence)},
            'intent': {'label': intent_label, 'confidence': float(intent_confidence)},
            'urgency': {'label': urgency_label, 'confidence': float(urgency_confidence)},
            'explanations': {
                'category': category_explanation,
                'intent': intent_explanation,
                'urgency': urgency_explanation
            },
            'success': True
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============= EXPLANATION VISUALIZATION FUNCTIONS =============

def create_explanation_chart(explanation, head_name):
    """Create interactive explanation chart using Plotly"""

    exp_data = explanation.as_list()
    if not exp_data:
        return None

    words = [item[0] for item in exp_data]
    importances = [item[1] for item in exp_data]

    # Create color map
    colors = ['#28a745' if imp > 0 else '#dc3545' for imp in importances]

    fig = go.Figure(data=[
        go.Bar(
            y=words,
            x=importances,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{imp:.3f}' for imp in importances],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=f'{head_name} Word Importance',
        xaxis_title='LIME Importance Score',
        yaxis_title='Words',
        height=400,
        margin=dict(l=100, r=50, t=50, b=50)
    )

    return fig

def highlight_text_with_explanations(text, explanation):
    """Create highlighted text based on LIME explanation"""

    exp_data = explanation.as_list()
    exp_dict = {word: importance for word, importance in exp_data}

    words = text.split()
    highlighted_words = []

    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())

        if clean_word in exp_dict:
            importance = exp_dict[clean_word]
            if importance > 0:
                # Positive importance - green
                intensity = min(abs(importance) * 2, 1.0)
                highlighted_words.append(f'<span style="background-color: rgba(40, 167, 69, {intensity}); padding: 2px 4px; border-radius: 3px; margin: 1px;">{word}</span>')
            else:
                # Negative importance - red
                intensity = min(abs(importance) * 2, 1.0)
                highlighted_words.append(f'<span style="background-color: rgba(220, 53, 69, {intensity}); padding: 2px 4px; border-radius: 3px; margin: 1px;">{word}</span>')
        else:
            highlighted_words.append(word)

    return ' '.join(highlighted_words)

# ============= HELPER FUNCTIONS =============

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_urgency_emoji(urgency):
    """Get emoji for urgency level"""
    emoji_map = {
        'Low': '🟢',
        'Medium': '🟡',
        'High': '🔴'
    }
    return emoji_map.get(urgency, '⚪')


# ============= STREAMLIT UI =============

def main():
    # Page configuration
    st.set_page_config(
        page_title="Banking AI Customer Service Assistant",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .explanation-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .explainability-badge {
        background-color: #17a2b8;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header with explainability badge
    st.markdown('''
    <h1 class="main-header">
        🏦 Banking AI Customer Service Assistant
        <span class="explainability-badge">XAI Powered</span>
    </h1>
    ''', unsafe_allow_html=True)

    # Load model components
    with st.spinner('Loading AI model and explainability components...'):
        model, tokenizer, label_encoders, model_info, explainer = load_model_components()

    # Sidebar with enhanced info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model Version:** V3 (XAI Enhanced)")
        st.write(f"**Explainability Method:** LIME")
        st.write(f"**Vocabulary Size:** {model_info.get('vocab_size', 'N/A')}")
        st.write(f"**Max Sequence Length:** {model_info.get('max_length', 'N/A')}")

        st.header("🎯 Model Capabilities")
        st.write("**Category Classification:**")
        st.write("• Card Services")
        st.write("• Account Services")
        st.write("• Transfer Services")
        st.write("• Support Services")

        st.header("🧠 Explainable AI Features")
        st.write("**🔍 LIME Explanations:**")
        st.write("• Word-level importance scoring")
        st.write("• Visual explanation charts")
        st.write("• Interactive text highlighting")
        st.write("• Natural language summaries")

        st.write("**Intent Recognition:** 77 intents")
        st.write("**Urgency Detection:** Low/Medium/High")

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("💬 Customer Query Analysis")

        # Text input
        user_input = st.text_area(
            "Enter customer query for AI analysis with explanations:",
            placeholder="Example: I urgently need to transfer money to my friend...",
            height=120,
            help="Enter a banking query and get AI predictions with detailed explanations"
        )

        # Explanation options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_explanations = st.checkbox("Show Detailed Explanations", value=True)
        with col_opt2:
            show_charts = st.checkbox("Show Explanation Charts", value=True)

        # Analysis button
        if st.button("🔍 Analyze Query", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner('Analyzing query and generating explanations...'):
                    # Make prediction with explanations
                    result = make_prediction_with_explanations(
                        user_input, model, tokenizer, label_encoders,
                        model_info['max_length'], explainer
                    )

                if result['success']:
                    st.success("Analysis completed with explanations!")

                    # Display basic predictions
                    st.subheader(" AI Predictions")
                    col_cat, col_intent, col_urgency = st.columns(3)

                    with col_cat:
                        confidence = result['category']['confidence']
                        conf_class = get_confidence_class(confidence)
                        st.markdown('<h4>📂 Category</h4>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size:18px; font-weight:bold;">{result["category"]["label"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<span class="{conf_class}">Confidence: {confidence:.1%}</span>', unsafe_allow_html=True)

                    with col_intent:
                        confidence = result['intent']['confidence']
                        conf_class = get_confidence_class(confidence)
                        st.markdown('<h4>🎯 Intent</h4>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size:18px; font-weight:bold;">{result["intent"]["label"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<span class="{conf_class}">Confidence: {confidence:.1%}</span>', unsafe_allow_html=True)

                    with col_urgency:
                        confidence = result['urgency']['confidence']
                        conf_class = get_confidence_class(confidence)
                        urgency_emoji = get_urgency_emoji(result['urgency']['label'])
                        st.markdown('<h4>⚡ Urgency</h4>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size:18px; font-weight:bold;">{urgency_emoji} {result["urgency"]["label"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<span class="{conf_class}">Confidence: {confidence:.1%}</span>', unsafe_allow_html=True)

                    if show_explanations:
                        # Explanation summary - FIXED VERSION
                        st.markdown("---")
                        st.subheader("🧠 AI Explanation Summary")

                        # Extract explanation data
                        category = result['category']['label']
                        intent = result['intent']['label']
                        urgency = result['urgency']['label']

                        cat_words = [word for word, _ in result['explanations']['category'].as_list()[:3]]
                        urgency_words = [word for word, _ in result['explanations']['urgency'].as_list()[:3]]



                        # Display explanation
                        st.info(f"""
                         **AI Decision Explanation**

                        **Category Classification:** The model classified this as **{category}** primarily based on keywords like *{', '.join(cat_words)}*.

                        **Intent Detection:** The specific intent was identified as **{intent}** based on the overall context and word patterns in your query.

                        **Urgency Assessment:** The urgency level was determined as **{urgency}** by analyzing words such as *{', '.join(urgency_words)}*.

                        💡 The highlighted words below show which parts of your text most influenced each decision.
                        """)

                        # Text highlighting for each head
                        st.subheader("🔍 Word Importance Highlighting")

                        tabs = st.tabs(["📂 Category", "🎯 Intent", "⚡ Urgency"])

                        with tabs[0]:
                            highlighted = highlight_text_with_explanations(
                                user_input, result['explanations']['category']
                            )
                            st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>{highlighted}</div>",
                                      unsafe_allow_html=True)
                            st.caption("🟢 Green: Promotes category prediction | 🔴 Red: Demotes category prediction")

                        with tabs[1]:
                            highlighted = highlight_text_with_explanations(
                                user_input, result['explanations']['intent']
                            )
                            st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>{highlighted}</div>",
                                      unsafe_allow_html=True)
                            st.caption("🟢 Green: Promotes intent prediction | 🔴 Red: Demotes intent prediction")

                        with tabs[2]:
                            highlighted = highlight_text_with_explanations(
                                user_input, result['explanations']['urgency']
                            )
                            st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>{highlighted}</div>",
                                      unsafe_allow_html=True)
                            st.caption("🟢 Green: Promotes urgency prediction | 🔴 Red: Demotes urgency prediction")

                    if show_charts:
                        # Interactive explanation charts
                        st.markdown("---")
                        st.subheader("📊 Interactive Explanation Charts")

                        chart_tabs = st.tabs(["📂 Category Chart", "🎯 Intent Chart", "⚡ Urgency Chart"])

                        with chart_tabs[0]:
                            fig = create_explanation_chart(result['explanations']['category'], "Category")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        with chart_tabs[1]:
                            fig = create_explanation_chart(result['explanations']['intent'], "Intent")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        with chart_tabs[2]:
                            fig = create_explanation_chart(result['explanations']['urgency'], "Urgency")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        # Recommendations
                        #recommendations = generate_recommendations(result)
                        #st.markdown("### 💡 Recommended Actions")
                        #for rec in recommendations:
                            #st.write(f"• {rec}")

                else:
                    st.error(f"❌ Error during analysis: {result['error']}")
            else:
                st.warning("Please enter a customer query to analyze.")

    with col2:
        st.header("📈 Model Performance")

        # Performance metrics
        metrics_data = {
            'Category': 0.87,
            'Intent': 0.90,
            'Urgency': 0.75,
            'Overall': 0.83
        }

        for metric, score in metrics_data.items():
            st.metric(f"{metric} Accuracy", f"{score:.1%}")

        st.markdown("---")

        # Explainability info
        st.header("🧠 Explainability Info")
        st.info("""
        **LIME (Local Interpretable Model-Agnostic Explanations)**

        • Explains individual predictions
        • Shows word-level importance
        • Model-agnostic approach
        • Provides trustworthy insights
        """)

        # Example queries
        st.header("💡 Try These Examples")
        example_queries = [
            "I urgently need to transfer money to my friend",
            "My credit card is not working at the ATM machine",
            "Can you please help me check my account balance?",
            "I want to apply for a new debit card immediately",
            "There's a suspicious transaction on my statement"
        ]

        for i, example in enumerate(example_queries):
            if st.button(f"Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.code(example)

# ============= RUN THE APP =============
if __name__ == "__main__":
    main()

