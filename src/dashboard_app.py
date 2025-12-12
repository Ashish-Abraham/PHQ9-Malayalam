
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
    return create_live_emotion_chart(state), check_live_alerts(state)

def create_dashboard():
    with gr.Blocks(theme=gr.themes.Soft(), title="Clinical Dashboard") as demo:
        # Header
        risk = get_risk_level(CURRENT_PATIENT["totalScore"], CURRENT_PATIENT["symptoms"]["Suicidal Ideation"])
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"# PHQ-9 Clinical Dashboard\n"
                           f"**Patient ID:** {CURRENT_PATIENT['id']} | **Age:** {CURRENT_PATIENT['age']} | **Gender:** {CURRENT_PATIENT['gender']}")
            with gr.Column(scale=1):
                gr.Markdown(f"# {CURRENT_PATIENT['totalScore']}\n"
                           f"**PHQ-9 Total Score**\n\n"
                           f"<span style='background-color:{risk['color']}; color:white; padding: 4px 8px; border-radius: 4px;'>{risk['level']} Risk</span>")

        # Alerts
        patterns = detect_patterns(CURRENT_PATIENT)
        if patterns:
             with gr.Group():
                gr.Markdown("### Response Pattern Alerts")
                for p in patterns:
                    gr.Markdown(f"**{p['message']}** (Type: {p['type']} | Priority: {p['priority']})")
        
        # LIVE UPDATES SECTION
        with gr.Group():
            gr.Markdown("## 🔴 Live Session Analysis")
            with gr.Row():
                live_emo_plot = gr.Plot(label="Live Emotions")
                live_alerts = gr.Markdown(label="Live Alerts")
            
            # Timer for polling
            timer = gr.Timer(value=2) # Update every 2 seconds
            timer.tick(update_dashboard, outputs=[live_emo_plot, live_alerts])

        # Charts Row 1
        with gr.Row():
            with gr.Column():
                symptom_plot = gr.Plot(create_symptom_chart(CURRENT_PATIENT))
                discordance = get_discordance(CURRENT_PATIENT)
                gr.Markdown(f"**Symptom Variance:** {discordance['variance']}\n{discordance['interpretation']}")
            
            with gr.Column():
                radar_plot = gr.Plot(create_radar_chart(CURRENT_PATIENT))
                gr.Info("Clinical Note: Patient shows elevated mood and energy symptoms compared to population baseline.")

        # Charts Row 2
        with gr.Row():
            with gr.Column():
                pop_plot = gr.Plot(create_population_chart(CURRENT_PATIENT))
            with gr.Column():
                gender_plot = gr.Plot(create_gender_chart(CURRENT_PATIENT))

        # External Stressors & Actions
        with gr.Row():
            with gr.Column():
                gr.Markdown("### External Stressor Profile")
                for factor, value in CURRENT_PATIENT["externalFactors"].items():
                    level = ["Good", "Average", "Bad", "Worst"][value]
                    color = "green" if value == 0 else "orange" if value == 1 else "red"
                    gr.Markdown(f"**{factor}**: {level} (Level {value})")
            
            with gr.Column():
                gr.Markdown(f"### Clinical Action Items\n"
                           f"- {risk['action']}\n"
                           f"- Review suicidal ideation response in detail with patient\n"
                           f"- Consider referral for financial counseling and academic support services\n"
                           f"- Schedule follow-up assessment in 2 weeks")
                           
    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(server_name="0.0.0.0", server_port=7861)
