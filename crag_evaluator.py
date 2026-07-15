import config
from app_init import llm_client

class CRAGEvaluator:
    def __init__(self, client):
        self.llm_client = client

    def evaluate_context(self, user_query, context_string):
        # 1. The Strict Logic Gate Persona

#        system_persona = """You are a strict, deterministic binary logic gate for an Intelligent Tutoring System.
#Your ONLY job is to determine if the provided Context contains sufficient, explicit facts to answer the User Query.
#- If the Context contains the answer, output exactly the word: CORRECT
#- If the Context is irrelevant, shallow, or missing the answer, output exactly the word: INCORRECT
#CRITICAL: You must output ONLY one single word. No punctuation, no explanations, no conversational text."""

        system_persona = """You are a strict retrieval evaluator for an Intelligent Tutoring System.
Your job is to measure Information Entailment. 
Apply this strict logical test:
"If a student read ONLY this provided Context and had no outside knowledge, would they have enough specific, factual information to accurately answer the User Query?"
- If YES (the context contains the actual definitions, mechanics, or data needed): output exactly CORRECT
- If NO (the context only mentions the topic by name, lacks the actual explanation, is structural metadata, or is irrelevant): output exactly INCORRECT
CRITICAL: You must output ONLY one single word. No punctuation, no explanations."""

        user_instruction = f"""
        User Query: {user_query}
        Context: {context_string}
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_persona},
                    {"role": "user", "content": user_instruction}
                ],
                temperature=0.0,  # 2. Zero creativity allowed
                max_tokens=10     # 3. Hard cutoff to prevent rambling
            )
            
            # Clean the output to ensure it's a perfect string match
            evaluator_decision = response.choices[0].message.content.strip().upper()
            
            # Failsafe: If the LLM somehow hallucinates something else, default to INCORRECT
            if "CORRECT" in evaluator_decision and "INCORRECT" not in evaluator_decision:
                return "CORRECT"
            else:
                return "INCORRECT"
                
        except Exception as e:
            print(f"Evaluator Error: {e}")
            # If the API crashes, fail safely by tripping the interlock
            return "INCORRECT"

# ==========================================
# QUICK TEST RIG
# ==========================================
if __name__ == "__main__":
    evaluator = CRAGEvaluator(llm_client)
    
    print("Testing the Interlock...")
    
    # Test 1: Good Data (Should pass)
    good_context = "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
    result_1 = evaluator.evaluate_context("What is machine learning?", good_context)
    print(f"Test 1 (Good Data): {result_1}")
    
    # Test 2: Garbage Data (Should trip the interlock)
    bad_context = "S.NO Title 1 Introduction to Machine learning 2 Feature Selection"
    result_2 = evaluator.evaluate_context("What is machine learning?", bad_context)
    print(f"Test 2 (Garbage Data): {result_2}")