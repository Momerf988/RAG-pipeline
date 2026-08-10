# V2 UPDATED (10-08-2026)
import config

# PHASE 3: LLM response generation
class LLMResponseProcessor:
    def __init__(self, llm_client, db_index, reranker_model):
        self.llm_client = llm_client
        self.index = db_index
        self.reranker = reranker_model

    def search_db(self, embedded_user_query, top_k = 20):
        search_result = self.index.query(
            vector = embedded_user_query,
            top_k = top_k,
            include_metadata = True)
        return search_result['matches']

    def rerank(self, user_prompt, matches, keep_top = 7):
        if not matches:
            return matches
        pairs = [(user_prompt, match['metadata']['line']) for match in matches]
        scores = self.reranker.predict(pairs)
        scored_matches = list(zip(scores, matches))
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        return [match for score, match in scored_matches[:keep_top]]

    # v2 fix (HANDOFF rule 5 / pre-flight gates: "stich_context dropped chunk IDs -- every
    # retrieval function returns chunk IDs"). Now returns chunk_id_list as a third value.
    # search_db() and rerank() are untouched.
    def stich_context(self, matches):
        context_list = [match['metadata']['line'] for match in matches]
        chunk_id_list = [match['id'] for match in matches]
        stiched_context = " ".join(context_list)
        return stiched_context, context_list, chunk_id_list

    def LLM_prompt(self, user_prompt, detail_level, context_string):
        system_persona = "You are a highly intelligent, helpful university teaching assistant."
        critical_rule = '''
        CRITICAL RULE: Your answer must come ENTIRELY from the Context below.
        - If the Context contains the answer, use ONLY facts from the Context.
        - If the Context only mentions the topic name without explaining it, that is NOT sufficient -- reply with the sorry message below.
        - If you find yourself using knowledge NOT visible in the Context (dates, version numbers, historical facts, extra detail), stop and reply with the sorry message instead.
        Reply EXACTLY with the following if the Context is insufficient:
        "Sorry, there are no relevant documents in the Database to answer your query. :("
        '''
        user_instruction = f'''
        Please answer the user's question. Provide the answer at {detail_level} level.
        NOTE: Remember to strictly follow this: {critical_rule}
        Context: {context_string}
        User prompt: {user_prompt}
        '''
        return system_persona, user_instruction

    # v2 fix (HANDOFF limitation: "Tutor T = 0.1" -> T = 0.0; and "No max_tokens; 6 items
    # lost" -> max_tokens set explicitly, finish_reason returned so the caller can assert it).
    def generate_response(self, system_persona, user_instruction):
        response = self.llm_client.chat.completions.create(
            model = config.LLM_MODEL_NAME,
            messages = [
                {"role": "system", "content": system_persona},
                {"role": "user", "content": user_instruction}
            ],
            temperature = config.TEMP_TUTOR,          # was 0.1 in V1
            max_tokens = config.TUTOR_MAX_TOKENS       # V1 set no limit at all
        )
        final_answer = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        return final_answer, finish_reason

    # v2 fix: propagates chunk_id_list, and hard-asserts finish_reason == "stop" (HANDOFF
    # rule 5: "finish_reason == 'stop' on every generation").
    def respond_to_user(self, embedded_user_query, user_prompt, detail_level):
        db_search_result = self.search_db(embedded_user_query)
        reranked_matches = self.rerank(user_prompt, db_search_result)
        stiched_context, context_list, chunk_id_list = self.stich_context(reranked_matches)
        if not stiched_context:
            return "Sorry, there are no relevant documents in the Database to answer your query. :(", [], []
        system_persona, user_instruction = self.LLM_prompt(user_prompt, detail_level, stiched_context)
        final_answer, finish_reason = self.generate_response(system_persona, user_instruction)
        assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
        return final_answer, context_list, chunk_id_list
