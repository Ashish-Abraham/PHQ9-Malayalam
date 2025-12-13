
import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import math

# --- Mock Data ---
CURRENT_PATIENT = {
    "id": "P-2847",
    "age": 24,
    "gender": "Female",
    "symptoms": {
        "Interest/Pleasure": 2,
        "Feeling Down": 3,
        "Sleep Issues": 2,
        "Fatigue": 3,
        "Appetite": 1,
        "Self-Worth": 2,
        "Concentration": 2,
        "Psychomotor": 1,
        "Suicidal Ideation": 1
    },
    "externalFactors": {
        "Sleep Quality": 2,  # Bad
        "Study Pressure": 2,  # Bad
        "Financial Pressure": 1  # Average
    },
    "totalScore": 17,
    "conversationMetrics": {
        "avgResponseTime": 4.2,
        "emotionalVariance": 0.72,
        "elaborationScore": 6.5
    }
}

POPULATION_DATA = [
    {"ageGroup": "18-22", "avgScore": 12.3, "count": 145, "q1": 8, "q3": 16},
    {"ageGroup": "23-25", "avgScore": 14.8, "count": 98, "q1": 10, "q3": 19},
    {"ageGroup": "26-30", "avgScore": 13.1, "count": 67, "q1": 9, "q3": 17},
    {"ageGroup": "31-35", "avgScore": 11.5, "count": 43, "q1": 7, "q3": 15},
]

GENDER_COMPARISON = [
    {"gender": "Female", "avgScore": 14.2, "count": 198},
    {"gender": "Male", "avgScore": 12.1, "count": 143},
    {"gender": "Other", "avgScore": 15.7, "count": 12}
]

# --- Logic & Analysis ---

def get_risk_level(score, suicidal_score):
    if suicidal_score >= 2 or score >= 20:
        return {"level": "High", "color": "#ef4444", "action": "Immediate clinical attention required"}
    if score >= 15:
        return {"level": "Moderate-High", "color": "#f59e0b", "action": "Clinical follow-up recommended within 1 week"}
    if score >= 10:
        return {"level": "Moderate", "color": "#eab308", "action": "Monitor and schedule follow-up"}
    return {"level": "Mild", "color": "#22c55e", "action": "Supportive intervention"}

def detect_patterns(patient):
    patterns = []
    symptoms = patient["symptoms"]
    
    # High suicidal ideation
    if symptoms.get("Suicidal Ideation", 0) >= 2 and patient["totalScore"] < 15:
        patterns.push({
            "type": "critical",
            "message": "⚠️ Elevated suicidal ideation despite moderate overall score",
            "priority": "high",
            "color": "red"
        })

    # Sleep + Fatigue
    if symptoms.get("Sleep Issues", 0) >= 2 and symptoms.get("Fatigue", 0) >= 2:
        patterns.append({
            "type": "clinical",
            "message": "Sleep-Fatigue cluster detected - consider sleep hygiene intervention",
            "priority": "medium",
             "color": "orange"
        })

    # Cognitive
    if symptoms.get("Concentration", 0) >= 2 and symptoms.get("Self-Worth", 0) >= 2:
        patterns.append({
            "type": "clinical",
            "message": "Cognitive-emotional pattern - may benefit from CBT",
            "priority": "medium",
             "color": "orange"
        })

    # External stressors
    external_sum = sum(patient["externalFactors"].values())
    if external_sum >= 5:
        patterns.append({
            "type": "contextual",
            "message": "High external stressors - consider resource referrals",
            "priority": "medium",
             "color": "blue"
        })
        
    return patterns

def get_discordance(patient):
    symptoms = list(patient["symptoms"].values())
    mean = sum(symptoms) / len(symptoms)
    variance = sum([pow(x - mean, 2) for x in symptoms]) / len(symptoms)
    std = math.sqrt(variance)
    
    return {
        "variance": round(variance, 2),
        "std": round(std, 2),
        "interpretation": "High variability - uneven symptom profile" if std > 0.9 else "Relatively consistent symptom severity"
    }

# --- Charts ---

def create_symptom_chart(patient):
    if not patient or not patient.get("symptoms"):
        # Return empty chart or placeholder
        fig = go.Figure()
        fig.update_layout(
            title="Symptom Profile (Waiting for data...)",
            yaxis_range=[0, 3],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    data = []
    for name, value in patient["symptoms"].items():
        data.append({
            "Symptom": name.replace("/", "/\n"),
            "Score": value,
            "Type": "Patient"
        })
        # Mock population avg
        data.append({
            "Symptom": name.replace("/", "/\n"),
            "Score": (hash(name) % 20) / 10 + 0.5, # Deterministic mock
            "Type": "Population Avg"
        })
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    patient_data = df[df["Type"] == "Patient"]
    pop_data = df[df["Type"] == "Population Avg"]
    
    fig.add_trace(go.Bar(
        x=patient_data["Symptom"],
        y=patient_data["Score"],
        name="Patient Score",
        marker_color='#3b82f6'
    ))
    
    fig.add_trace(go.Bar(
        x=pop_data["Symptom"],
        y=pop_data["Score"],
        name="Population Avg",
        marker_color='#94a3b8'
    ))
    
    fig.update_layout(
        title="Symptom Profile Analysis",
        yaxis_range=[0, 3],
        barmode='group'
    )
    return fig

def create_radar_chart(patient):
    if not patient or not patient.get("symptoms"):
         return go.Figure().update_layout(title="Symptom Domain (Waiting for data...)", polar=dict(radialaxis=dict(visible=False)))
         
    symptoms = patient["symptoms"]
    categories = [
        {"category": "Mood", "patient": (symptoms["Feeling Down"] + symptoms["Interest/Pleasure"]) / 2, "pop": 1.6},
        {"category": "Sleep", "patient": symptoms["Sleep Issues"], "pop": 1.8},
        {"category": "Energy", "patient": symptoms["Fatigue"], "pop": 1.9},
        {"category": "Cognitive", "patient": (symptoms["Concentration"] + symptoms["Psychomotor"]) / 2, "pop": 1.4},
        {"category": "Self-Perception", "patient": symptoms["Self-Worth"], "pop": 1.5},
        {"category": "Somatic", "patient": symptoms["Appetite"], "pop": 1.3}
    ]
    
    cats = [c["category"] for c in categories]
    pat_vals = [c["patient"] for c in categories]
    pop_vals = [c["pop"] for c in categories]
    
    # Close the loop
    cats.append(cats[0])
    pat_vals.append(pat_vals[0])
    pop_vals.append(pop_vals[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=pat_vals,
        theta=cats,
        fill='toself',
        name='Patient',
        line_color='#3b82f6'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=pop_vals,
        theta=cats,
        fill='toself',
        name='Population Avg',
        line_color='#94a3b8',
        opacity=0.5
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3]
            )),
        title="Symptom Domain Comparison"
    )
    return fig

def create_population_chart(patient):
    if not patient or "totalScore" not in patient:
         return go.Figure().update_layout(title="Population Benchmark (Waiting for data...)")

    df = pd.DataFrame(POPULATION_DATA)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["ageGroup"],
        y=df["avgScore"],
        name="Population Avg",
        marker_color='#10b981'
    ))
    
    # Add patient line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=patient["totalScore"],
        x1=3.5,
        y1=patient["totalScore"],
        line=dict(
            color="#ef4444",
            width=2,
            dash="dashdot",
        ),
    )
    
    fig.update_layout(
        title="Age Group Benchmarking",
        yaxis_range=[0, 25]
    )
    return fig

def create_gender_chart(patient):
    if not patient or "gender" not in patient:
        return go.Figure().update_layout(title="Gender Comparison (Waiting for data...)")
        
    df = pd.DataFrame(GENDER_COMPARISON)
    colors = ['#ef4444' if g == patient['gender'] else '#8b5cf6' for g in df['gender']]
    
    fig = go.Figure(go.Bar(
        x=df["avgScore"],
        y=df["gender"],
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Gender-Based Comparison",
        xaxis_range=[0, 20]
    )
    return fig

from src.shared_state import get_dashboard_state

def create_live_emotion_chart(state):
    """Create chart from live state."""
    if not state or not state.get("top_emotions"):
        return go.Figure().update_layout(title="Waiting for live data...")
        
    # Get top 3
    emotions = state["top_emotions"]
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    
    fig = go.Figure(go.Bar(
        x=[e[0] for e in sorted_emotions],
        y=[e[1] for e in sorted_emotions],
        marker_color='#f43f5e'
    ))
    fig.update_layout(title="Live User Emotions (Top 3)", yaxis_title="Count")
    return fig

def check_live_alerts(state):
    """Get live alerts text."""
    if not state: return ""
    alerts = state.get("suicide_risk", {}).get("alerts", [])
    if not alerts: return ""
    
    html = "### ⚠️ LIVE WARNINGS<br>"
    for alert in alerts[-3:]: # Show last 3
        html += f"<div style='background-color:#fee2e2; border-left: 4px solid #ef4444; padding: 10px; margin-bottom: 5px; color: #b91c1c;'><b>{alert['message']}</b></div>"
    return html

def update_dashboard():
    """Poll shared state and update live components."""
    state = get_dashboard_state()
    
    # 1. Charts
    live_emo_chart = create_live_emotion_chart(state)
    live_risk = check_live_alerts(state)
    
    # 2. Patient Info Header (Update in case login happened after dashboard load)
    header_md = ""
    patient = state.get("patient") if state else None
    if patient:
        header_md = f"# PHQ-9 Clinical Dashboard\n**Patient ID:** {patient.get('id', 'N/A')} | **Name:** {patient.get('name', 'N/A')} | **Age:** {int(patient.get('age', 0))} | **Gender:** {patient.get('gender', 'N/A')}"
    else:
        header_md = f"# PHQ-9 Clinical Dashboard\n**Patient ID:** Waiting... | **Status:** No active session"

    # 3. Score Header
    # Calculate score from real symptoms if available
    symptoms = state.get("symptoms", {}) if state else {}
    if symptoms:
        total_score = sum(symptoms.values())
        risk = get_risk_level(total_score, symptoms.get("Suicidal Ideation", 0))
        score_md = f"# {int(total_score)}\n**PHQ-9 Total Score**\n\n<span style='background-color:{risk['color']}; color:white; padding: 4px 8px; border-radius: 4px;'>{risk['level']} Risk</span>"
    else:
        score_md = "# --\n**PHQ-9 Total Score**"

    # 5. Other Charts
    # Ensure patient object has 'totalScore' or calculate it
    if symptoms:
        # Default missing keys to 0 for radar chart safety
        SAFE_KEYS = [
            "Interest/Pleasure", "Feeling Down", "Sleep Issues", "Fatigue",
            "Appetite", "Self-Worth", "Concentration", "Psychomotor", "Suicidal Ideation"
        ]
        safe_symptoms = {k: symptoms.get(k, 0) for k in SAFE_KEYS}
        
        patient_for_charts = {
            "symptoms": safe_symptoms, 
            "totalScore": sum(symptoms.values()), 
            "gender": patient.get("gender") if patient else "Female"
        }
    else:
        patient_for_charts = None
        
    symptom_fig = create_symptom_chart(patient_for_charts)
    radar_fig = create_radar_chart(patient_for_charts)
    pop_fig = create_population_chart(patient_for_charts)
    gender_fig = create_gender_chart(patient_for_charts)
    
    # 6. External Stressors
    ext_factors = state.get("external_factors", {}) if state else {}
    if ext_factors:
        ext_md = "### External Stressor Profile\n"
        for factor, value in ext_factors.items():
            level = ["Good", "Average", "Bad", "Worst"][value] if value < 4 else "Unknown"
            color = "green" if value == 0 else "orange" if value == 1 else "red"
            ext_md += f"<span style='color:{color}'>**{factor}**: {level} (Level {value})</span><br>"
    else:
        ext_md = "### External Stressor Profile\nWaiting for data..."

    # 7. Action Items
    if symptoms:
        # Re-calc risk for action items
        total_score = sum(symptoms.values())
        risk_obj = get_risk_level(total_score, symptoms.get("Suicidal Ideation", 0))
        act_md = f"### Clinical Action Items\n- **{risk_obj['level']} Risk Plan**: {risk_obj['action']}\n- Review suicidal ideation response in detail.\n- Schedule follow-up."
    else:
        act_md = "### Clinical Action Items\nWaiting for clinical data..."
    
    return live_emo_chart, live_risk, header_md, score_md, symptom_fig, radar_fig, pop_fig, gender_fig, ext_md, act_md

def create_dashboard():
    with gr.Blocks(theme=gr.themes.Soft(), title="Clinical Dashboard") as demo:
        # Header
        with gr.Row():
            with gr.Column(scale=3):
                header = gr.Markdown("# PHQ-9 Clinical Dashboard\nWaiting for patient data...")
            with gr.Column(scale=1):
                score_display = gr.Markdown("# --\n**PHQ-9 Total Score**")

        # LIVE UPDATES SECTION
        with gr.Group():
            gr.Markdown("## 🔴 Live Session Analysis")
            with gr.Row():
                live_emo_plot = gr.Plot(label="Live Emotions")
                live_alerts = gr.Markdown(label="Live Alerts")

        # Charts Row 1
        with gr.Row():
            with gr.Column():
                symptom_plot = gr.Plot(value=create_symptom_chart(None), label="Symptom Profile")
            
            with gr.Column():
                radar_plot = gr.Plot(value=create_radar_chart(None), label="Symptom Radar")
                gr.Info("Clinical Note: Radar chart shows domain-level analysis.")

        # Charts Row 2
        with gr.Row():
            with gr.Column():
                pop_plot = gr.Plot(value=create_population_chart(None), label="Population Benchmark")
            with gr.Column():
                gender_plot = gr.Plot(value=create_gender_chart(None), label="Gender Comparison")

        # External Stressors & Actions
        with gr.Row():
            with gr.Column():
                ext_md = gr.Markdown("### External Stressor Profile\nWaiting for data...")
            
            with gr.Column():
                act_md = gr.Markdown("### Clinical Action Items\nWaiting for data...")

        # Timer for polling
        timer = gr.Timer(value=2) # Update every 2 seconds
        
        # Update everything
        timer.tick(
            update_dashboard, 
            outputs=[live_emo_plot, live_alerts, header, score_display, symptom_plot, radar_plot, pop_plot, gender_plot, ext_md, act_md]
        )
                            
    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(server_name="0.0.0.0", server_port=7861)
