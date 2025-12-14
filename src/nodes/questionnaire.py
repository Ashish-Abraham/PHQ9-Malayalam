from state import AgentState
from utils.llm import get_llm
from langchain_core.messages import AIMessage
import json

PHQ9_QUESTIONS_MAL = [
    "കാര്യങ്ങൾ ചെയ്യാൻ താല്പര്യക്കുറവോ സന്തോഷമില്ലായ്മയോ അനുഭവപ്പെടുന്നുണ്ടോ?",
    "വിഷമമോ, വിഷാദമോ, അല്ലെങ്കിൽ പ്രതീക്ഷയറ്റ അവസ്ഥയോ അനുഭവപ്പെടുന്നുണ്ടോ?",
    "ഉറങ്ങാൻ ബുദ്ധിമുട്ട്, അല്ലെങ്കിൽ ഇടയ്ക്കിടെ ഉറക്കം ഉണരുക, അല്ലെങ്കിൽ അമിതമായി ഉറങ്ങുക?",
    "ക്ഷീണമോ ഊർജ്ജമില്ലായ്മയോ അനുഭവപ്പെടുന്നുണ്ടോ?",
    "വിശപ്പില്ലായ്മയോ അമിതമായി ഭക്ഷണം കഴിക്കുന്നതോ?",
    "സ്വയം വെറുപ്പ് തോന്നുകയോ, താനൊരു പരാജയമാണെന്ന് തോന്നുകയോ, അല്ലെങ്കിൽ കുടുംബത്തിന് അപമാനമാണെന്ന് തോന്നുകയോ ചെയ്യുന്നുണ്ടോ?",
    "പത്രം വായിക്കുമ്പോഴോ ടിവി കാണുമ്പോഴോ ശ്രദ്ധ കേന്ദ്രീകരിക്കാൻ ബുദ്ധിമുട്ട് അനുഭവപ്പെടുന്നുണ്ടോ?",
    "മറ്റുള്ളവർ ശ്രദ്ധിക്കത്തക്കവിധം സാവധാനം സംസാരിക്കുകയോ നടക്കുകയോ ചെയ്യുക? അല്ലെങ്കിൽ അമിതമായി അസ്വസ്ഥനായി എപ്പോഴും ചലിച്ചുകൊണ്ടിരിക്കുക?",
    "മരിക്കുന്നതാണ് നല്ലതെന്നോ, സ്വയം വേദനിപ്പിക്കുന്നതിനെക്കുറിച്ചോ ഉള്ള ചിന്തകൾ?"
]

PHQ9_QUESTIONS_EN = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
    "Thoughts that you would be better off dead, or of hurting yourself?"
]

PHQ9_QUESTIONS = PHQ9_QUESTIONS_MAL # Backward compatibility if needed, but logic should switch

def get_question(index, language="English"):
    if language == "Malayalam":
        return PHQ9_QUESTIONS_MAL[index]
    return PHQ9_QUESTIONS_EN[index]

def questionnaire_node(state: AgentState):
    """
    Node for administering the PHQ-9 questionnaire.
    """
    llm = get_llm()
    messages = state['messages']
    current_index = state.get('current_question_index', 0)
    responses = state.get('phq9_responses', {})
    language = state.get("language", "English")
    
    questions = PHQ9_QUESTIONS_MAL if language == "Malayalam" else PHQ9_QUESTIONS_EN
    
    # If we have a user response (i.e., not the first time entering), process it
    if messages and messages[-1].type == 'human':
        from src.utils.message_utils import get_message_text
        last_response = get_message_text(messages[-1])
        
        # 1. Check for irrelevance/ambiguity and score
        # 1. Check for irrelevance/ambiguity and score
        
        if language == "Malayalam":
            scoring_prompt = f"""
            The user was asked: "{questions[current_index]}"
            The user answered: "{last_response}"
            
            The user is answering in Malayalam.
            
            Task:
            1. Determine if the answer is relevant to the question.
            2. If relevant, map it to a score based on frequency:
               - 0 : Not at all (ഒട്ടും ഇല്ല)
               - 1 : Several days (ചില ദിവസങ്ങളിൽ)
               - 2 : More than half the days (പകുതിയിലധികം ദിവസങ്ങളിൽ)
               - 3 : Nearly every day (മിക്കവാറും എല്ലാ ദിവസവും)
               
               Note: Look for Malayalam phrases indicating these frequencies.
               
            3. If ambiguous or irrelevant, indicate that.
            
            Output JSON format (do NOT include markdown formatting like ```json ... ```):
            {{
                "is_relevant": bool,
                "is_ambiguous": bool,
                "score": int (0-3, or null if not scorable),
                "clarification_needed": bool
            }}
            """
        else:
            scoring_prompt = f"""
            The user was asked: "{questions[current_index]}"
            The user answered: "{last_response}"
            
            Task:
            1. Determine if the answer is relevant to the question.
            2. If relevant, map it to a score: 0 (Not at all), 1 (Several days), 2 (More than half the days), 3 (Nearly every day).
            3. If ambiguous or irrelevant, indicate that.
            
            Output JSON format (do NOT include markdown formatting like ```json ... ```):
            {{
                "is_relevant": bool,
                "is_ambiguous": bool,
                "score": int (0-3, or null if not scorable),
                "clarification_needed": bool
            }}
            """
        
        try:
            analysis_response = llm.invoke([{"role": "system", "content": scoring_prompt}])
            content = analysis_response.content.strip()
            
            # Use robust parsing with regex to find the JSON block
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    analysis = json.loads(json_str)
                except json.JSONDecodeError as e:
    
                    # Try to clean up potential markdown code blocks inside the match if any (rare but possible)
                    clean_json = json_str.replace('```json', '').replace('```', '').strip()
                    analysis = json.loads(clean_json)
            else:
                 # Fallback to parser if regex fails (unlikely if JSON is present)
                from langchain_core.output_parsers import JsonOutputParser
                parser = JsonOutputParser()
                analysis = parser.parse(content)

            if analysis.get('is_relevant') and not analysis.get('is_ambiguous') and analysis.get('score') is not None:
                # Valid response
                responses[current_index] = analysis['score']
                current_index += 1
                state['phq9_responses'] = responses
                state['current_question_index'] = current_index
                
                from src.shared_state import update_symptoms
                # Questions mapping for dashboard keys
                DASHBOARD_KEYS = [
                    "Interest/Pleasure", "Feeling Down", "Sleep Issues", "Fatigue",
                    "Appetite", "Self-Worth", "Concentration", "Psychomotor", "Suicidal Ideation"
                ]
                
                # Build dashboard-friendly dict
                dash_symptoms = {}
                for idx, score in responses.items():
                    if idx < len(DASHBOARD_KEYS):
                        dash_symptoms[DASHBOARD_KEYS[idx]] = score
                
                update_symptoms(dash_symptoms)
                # ------------------------
            else:
                # Invalid/Ambiguous response
                # Generate a clarification request
                if language == "Malayalam":
                    clarification_prompt = f"""
                    The user's response "{last_response}" to the question "{questions[current_index]}" was ambiguous or irrelevant.
                    The conversation is in Malayalam.
                    Politely ask them to clarify it as 'ഒട്ടും ഇല്ല' (Not at all), 'ചില ദിവസങ്ങളിൽ' (Several days), 'പകുതിയിലധികം ദിവസങ്ങളിൽ' (More than half the days), or 'മിക്കവാറും എല്ലാ ദിവസവും' (Nearly every day) or bring them back to the topic.
                    Reply in Malayalam.
                    """
                else:
                    clarification_prompt = f"""
                    The user's response "{last_response}" to the question "{questions[current_index]}" was ambiguous or irrelevant.
                    Politely ask them to clarify it as 'Not at all', 'Several days', 'More than half the days', 'Nearly every day' or bring them back to the topic. 
                    """
                clarification = llm.invoke([{"role": "system", "content": clarification_prompt}])
                return {"messages": [clarification], "phase": "questionnaire"}
                
        except Exception as e:
            # Fallback for JSON parsing errors
            print(f"Error parsing scoring: {e}")
            print(f"Error parsing scoring: {e}")
            fallback_msg = "ക്ഷമിക്കണം, എനിക്ക് അത് മനസ്സിലായില്ല." if language == "Malayalam" else "I didn't quite catch that."
            return {"messages": [AIMessage(content=fallback_msg)], "phase": "questionnaire"}

    # Check if we are done
    if current_index >= len(questions):
        msg = "ഈ ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകിയതിന് നന്ദി. നിങ്ങളുടെ അവസ്ഥ നന്നായി മനസ്സിലാക്കാൻ എനിക്ക് കുറച്ച് ചോദ്യങ്ങൾ കൂടി ചോദിക്കാനുണ്ട്." if language == "Malayalam" else "Thank you for answering those questions. I have just a couple more questions to better understand your situation."
        return {"phase": "completed_phq9", "messages": [AIMessage(content=msg)], "phq9_responses": responses, "current_question_index": current_index} # Transition signal

    # Ask the next question
    question = questions[current_index]
    return {"messages": [AIMessage(content=question)], "current_question_index": current_index, "phq9_responses": responses, "phase": "questionnaire"}
