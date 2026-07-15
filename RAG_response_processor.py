import config

# PHASE 3: LLM response generation
class LLMResponseProcessor:
    def __init__(self, llm_client, db_index):
        self.llm_client = llm_client
        self.index = db_index

    def search_db(self, embedded_user_query, top_k = 3):
        search_result = self.index.query(
            vector = embedded_user_query, 
            top_k = 3, 
            include_metadata = True)
        matches = search_result['matches']
        if not matches:
            matches = []          # why not ''?
        return search_result['matches']
    
    def stich_context(self, matches):
        context_list = [match['metadata']['line'] for match in matches]
        stiched_context = " ".join(context_list)
        return stiched_context, context_list
    
    def LLM_prompt(self, user_prompt, detail_level, context_string):
        system_persona = "You are a highly intelligent, helpful university teaching assistant."
        critical_rule = '''
        CRITICAL RULE: You must base your answer EXCLUSIVELY on the Context provided below. 
        If the Context does not contain the necessary information to answer the question, 
        do NOT guess or make up an analogy. Reply EXACTLY with: 
        "Sorry, there are no relevant documents in the Database to answer your query. :("
        '''
        user_instruction = f'''
        Please answer the user's question. Provide the answer at {detail_level} level.
        NOTE: Remember to strictly follow this: {critical_rule}
        Context: {context_string}
        User prompt: {user_prompt} 
        '''
        return system_persona, user_instruction
    
    def generate_response(self, system_persona, user_instruction):
        response = self.llm_client.chat.completions.create(
            model = config.LLM_MODEL_NAME,
            messages = [
                {"role": "system", "content": system_persona},
                {"role": "user", "content": user_instruction}
            ],
            temperature = 0.1
        )
        return response.choices[0].message.content
    
    def respond_to_user(self, embedded_user_query, user_prompt, detail_level):
        db_search_result = self.search_db(embedded_user_query)
        stiched_context, context_list = self.stich_context(db_search_result)
        if not stiched_context:
            return "Sorry, there are no relevant documents in the Database to answer your query. :("
        system_persona, user_instruction = self.LLM_prompt(user_prompt, detail_level, stiched_context)
        final_answer = self.generate_response(system_persona, user_instruction)
        return final_answer, context_list
