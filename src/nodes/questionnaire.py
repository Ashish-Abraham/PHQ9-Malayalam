from state import AgentState
from utils.llm import get_llm
from langchain_core.messages import AIMessage
import json

PHQ9_QUESTIONS = [
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

def questionnaire_node(state: AgentState):
    """
    Node for administering the PHQ-9 questionnaire.
    """
    llm = get_llm()
    messages = state['messages']
    current_index = state.get('current_question_index', 0)
    responses = state.get('phq9_responses', {})
    
    # If we have a user response (i.e., not the first time entering), process it
    if messages and messages[-1].type == 'human':
        from src.utils.message_utils import get_message_text
        last_response = get_message_text(messages[-1])
        
        # 1. Check for irrelevance/ambiguity and score
        scoring_prompt = f"""
        The user was asked: "{PHQ9_QUESTIONS[current_index]}"
        The user answered: "{last_response}"
        
        Task:
        1. Determine if the answer is relevant to the question.
        2. If relevant, map it to a score: 0 (Not at all), 1 (Several days), 2 (More than half the days), 3 (Nearly every day).
        3. If ambiguous or irrelevant, indicate that and ask .
        
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
            
            # Use robust parsing
            from langchain_core.output_parsers import JsonOutputParser
            parser = JsonOutputParser()
            try:
                analysis = parser.parse(content)
            except Exception:
                # Fallback: naive strip if parser fails (though parser handles backticks usually)
                cleaned = content.replace('```json', '').replace('```', '').strip()
                analysis = json.loads(cleaned)

            if analysis.get('is_relevant') and not analysis.get('is_ambiguous') and analysis.get('score') is not None:
                # Valid response
                responses[current_index] = analysis['score']
                current_index += 1
                state['phq9_responses'] = responses
                state['current_question_index'] = current_index
            else:
                # Invalid/Ambiguous response
                # Generate a clarification request
                clarification_prompt = f"""
                The user's response "{last_response}" to the question "{PHQ9_QUESTIONS[current_index]}" was ambiguous or irrelevant.
                Politely ask them to clarify it as 'Not at all', 'Several days', 'More than half the days', 'Nearly every day' or bring them back to the topic.
                """
                clarification = llm.invoke([{"role": "system", "content": clarification_prompt}])
                return {"messages": [clarification], "phase": "questionnaire"}
                
        except Exception as e:
            # Fallback for JSON parsing errors
            print(f"Error parsing scoring: {e}")
            return {"messages": [AIMessage(content="I didn't quite catch that. Could you please answer with 'Not at all', 'Several days', 'More than half the days', or 'Nearly every day'?")], "phase": "questionnaire"}

    # Check if we are done
    if current_index >= len(PHQ9_QUESTIONS):
        return {"phase": "completed_phq9", "messages": [AIMessage(content="Thank you for answering those questions. I have just a couple more questions to better understand your situation.")]} # Transition signal

    # Ask the next question
    question = PHQ9_QUESTIONS[current_index]
    return {"messages": [AIMessage(content=question)], "current_question_index": current_index, "phq9_responses": responses, "phase": "questionnaire"}
