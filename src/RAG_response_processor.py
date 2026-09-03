import config


class LLMResponseProcessor:
    def __init__(self, llm_client, db_index, reranker_model):
        self.llm_client = llm_client
        self.index = db_index
        self.reranker = reranker_model

    def search_db(self, embedded_user_query, top_k=20):
        search_result = self.index.query(
            vector=embedded_user_query,
            top_k=top_k,
            include_metadata=True)
        return search_result['matches']

    # keep_top=15 is the settled value, decided by ablation rather than guessed: a
    # span-recall sweep across k showed recall still climbing well past the old default of
    # 7, and a follow-up quality check (real generation + RAGAS scoring at k=7/15/20) showed
    # k=15 giving the best faithfulness and answer relevancy of the three, with k=20 having
    # higher raw recall but worse generated-answer quality -- more retrieved text just
    # diluting what the tutor actually uses. Both ablations live in this repo's numbered
    # pipeline scripts, with keep_top_ablation_summary.csv holding the final numbers.
    def rerank(self, user_prompt, matches, keep_top=15):
        if not matches:
            return matches
        pairs = [(user_prompt, match['metadata']['line']) for match in matches]
        scores = self.reranker.predict(pairs)
        scored_matches = list(zip(scores, matches))
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        return [match for score, match in scored_matches[:keep_top]]

    def stich_context(self, matches):
        context_list = [match['metadata']['line'] for match in matches]
        chunk_id_list = [match['id'] for match in matches]
        stiched_context = " ".join(context_list)
        return stiched_context, context_list, chunk_id_list

    # System A's prompt. Deliberately has no sufficiency judgement of its own -- plain RAG,
    # by design, has no retrieval-quality gate at all, it generates from whatever was
    # retrieved, good or bad. A dedicated "is this context good enough?" check with
    # abstention is specifically what System B's CRAG interlock adds; giving System A that
    # same judgement would blur the comparison this whole project exists to make. The only
    # rule kept here is basic grounding hygiene -- stay faithful to the given context, no
    # outside facts -- which is standard RAG practice, not a CRAG contribution.
    def LLM_prompt(self, user_prompt, detail_level, context_string):
        system_persona = "You are a highly intelligent, helpful university teaching assistant."
        grounding_rule = '''
        GROUNDING RULE: Answer the user's question using ONLY facts, definitions, and
        explanations that are actually present in the Context below.
        - Do not add outside knowledge (dates, version numbers, historical facts, or extra
          detail) that is not visible in the Context.
        - Answer as accurately and completely as you can using what IS in the Context, even
          if it only partially covers the question.
        '''
        user_instruction = f'''
        Please answer the user's question. Provide the answer at {detail_level} level.
        NOTE: Remember to strictly follow this: {grounding_rule}
        Context: {context_string}
        User prompt: {user_prompt}
        '''
        return system_persona, user_instruction

    # System B's prompt, used only after CRAGEvaluator has already judged the context
    # CORRECT or usably AMBIGUOUS -- that judgement IS System B's sufficiency check, so this
    # prompt doesn't repeat it. Asking the tutor to independently re-judge sufficiency here
    # would let it silently override CRAG's own routing decision, which defeats the point of
    # having the evaluator in the first place. This keeps the same grounding rule as
    # LLM_prompt (answer only from the given context) but explicitly asks for the best answer
    # obtainable from partial context rather than refusing, since the whole reason CRAG has an
    # AMBIGUOUS verdict is to allow that instead of over-rejecting weak-but-usable context.
    # A genuine "nothing usable" refusal is handled entirely in the routing logic upstream of
    # this call and never reaches here.
    def LLM_prompt_crag(self, user_prompt, detail_level, context_string):
        system_persona = "You are a highly intelligent, helpful university teaching assistant."
        grounding_rule = '''
        GROUNDING RULE: Your answer must come ENTIRELY from the Context below.
        - Use ONLY facts from the Context. Do not use knowledge NOT visible in the Context
          (dates, version numbers, historical facts, extra detail).
        - A retrieval-quality check has already approved this context before you were asked
          to respond -- do not refuse to answer.
        - If the Context only partially covers the question, answer as accurately and
          completely as you can from what IS there, and briefly note if some aspect isn't
          covered by the given material.
        '''
        user_instruction = f'''
        Please answer the user's question. Provide the answer at {detail_level} level.
        NOTE: Remember to strictly follow this: {grounding_rule}
        Context: {context_string}
        User prompt: {user_prompt}
        '''
        return system_persona, user_instruction

    # Every generation call in both systems goes through here, interactive and batch alike.
    # Returns finish_reason alongside the answer so callers can hard-assert on "stop" rather
    # than silently accepting a truncated response. If the first attempt gets cut off at
    # TUTOR_MAX_TOKENS, this retries once with a modest bump (not a large multiple -- the
    # account's real ceiling is 6000 TPM combined prompt+completion, and keep_top=15 already
    # puts a few thousand tokens of context into every call, so a big multiple risks a 413
    # instead of fixing the truncation).
    def generate_response(self, system_persona, user_instruction):
        messages = [
            {"role": "system", "content": system_persona},
            {"role": "user", "content": user_instruction}
        ]
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=messages,
            temperature=config.TEMP_TUTOR,
            max_tokens=config.TUTOR_MAX_TOKENS
        )
        final_answer = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            retry_budget = config.TUTOR_MAX_TOKENS + 512
            print(f"  NOTE: response truncated at {config.TUTOR_MAX_TOKENS} tokens, "
                  f"retrying once with max_tokens={retry_budget}...")
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=config.TEMP_TUTOR,
                max_tokens=retry_budget
            )
            final_answer = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

        return final_answer, finish_reason

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
