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

    # V2 work -- keep_top raised 7 -> 15 per 09_tune_retrieval.py (span-recall) +
    # 10_tune_keep_top_quality.py (RAGAS quality ablation, n=24, System A). k=15 clears the
    # pre-declared +0.05 recall materiality bar by a wide margin AND wins on faithfulness
    # (0.476) and answer_relevancy (0.668) vs both k=7 and k=20 -- k=20 has higher raw
    # recall but measurably worse generated-answer quality (dilution), so it was rejected
    # despite the bigger recall number. See keep_top_ablation_summary.csv for full figures.
    def rerank(self, user_prompt, matches, keep_top = 15):
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

    # V2.5: removed the independent refusal judgment ("if insufficient, say sorry") from this
    # prompt. Per the CRAG paper (Yan et al., 2024) and the original RAG architecture (Lewis
    # et al., 2020) it builds on, plain RAG has no retrieval-quality gate at all -- it
    # generates from whatever was retrieved, good or bad. A dedicated sufficiency check with
    # abstention is specifically CRAG's contribution, not a standard RAG capability. This
    # method previously had its own bundled, single-shot "is this enough? -> refuse" judgment
    # baked into the same call that writes the answer -- a deliberate, disclosed anti-
    # hallucination safety addition from earlier in the project (already documented via the
    # "disguised refusal" investigation), but one that blurred the A-vs-B contrast this whole
    # study exists to measure. Fixed by keeping only the grounding rule (stay faithful to the
    # given Context, no outside facts -- standard RAG hygiene, not CRAG's contribution) and
    # dropping the sufficiency judgment and canned refusal message entirely. System A will now
    # do its best to answer from whatever context it's given, however weak -- this is the
    # textbook vanilla-RAG failure mode CRAG exists to correct, and demonstrating it (e.g. on
    # off-topic or irrelevant queries) is itself useful evidence for the comparison.
    # Verified before making this change: zero of the 72 keep_top-ablation rows (k=7/15/20)
    # ever triggered the old refusal string, so this does not invalidate the keep_top=15
    # decision or require rerunning that ablation. The hard `if not stiched_context` guard in
    # respond_to_user() below is untouched -- that's a code-level empty-input guard (can't
    # call the LLM with zero context), not a sufficiency judgment, and stays regardless.
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

    # V2 work -- CRAG-specific generation prompt, used ONLY by System B (system_b_main.py /
    # generate_evaluation_dataset_B.py). System A keeps using LLM_prompt above, unchanged.
    #
    # Why this exists: by the time System B calls this, CRAGEvaluator has already judged the
    # context CORRECT or AMBIGUOUS-and-worth-attempting -- that IS the sufficiency check for
    # System B. LLM_prompt's critical_rule asks the tutor to independently re-judge
    # sufficiency and refuse if it disagrees, which lets the tutor silently override CRAG's
    # own routing decision (found live during manual testing: CRAG said "usable", tutor
    # refused anyway). This method keeps the anti-hallucination grounding rule (answer only
    # from the given context, no outside facts) but removes the duplicate refusal check, and
    # explicitly asks for the best answer obtainable from partial context instead of bailing
    # out -- since CRAG's whole purpose in adding the AMBIGUOUS verdict was to allow exactly
    # that instead of over-rejecting. CRAG's genuine "nothing usable" refusal (both retrieval
    # attempts INCORRECT) is handled entirely in the routing code and never reaches here.
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

    # v2 fix (HANDOFF limitation: "Tutor T = 0.1" -> T = 0.0; and "No max_tokens; 6 items
    # lost" -> max_tokens set explicitly, finish_reason returned so the caller can assert it).
    #
    # V2.8: added a self-healing retry on truncation, fixed here (the one shared method every
    # generation call goes through -- System A's respond_to_user, System B's direct and
    # rewritten paths, interactive and batch alike) rather than in each caller separately.
    # Found live during the overnight run: finish_reason == "length" on row 34/150 even after
    # raising TUTOR_MAX_TOKENS to 2048, because keep_top=15 gives the tutor more to work with
    # on "elaborate" answers. Rather than have the batch scripts' AssertionError-skip logic
    # (see generate_evaluation_dataset*.py) be the first line of defence -- which means losing
    # rows from the final dataset silently unless someone notices the SKIPPED lines in a long
    # log -- this retries ONCE at 4x the configured budget before ever returning a truncated
    # result. The model supports up to 131,072 completion tokens (Groq's limit for
    # llama-3.1-8b-instant), so 4x headroom on a 2048 base is nowhere near any real ceiling.
    # The batch scripts' skip-on-AssertionError logic stays in place as a last-resort safety
    # net for the (should be near-impossible) case where even this doesn't produce finish_
    # reason == "stop" -- so no row silently vanishes without at least a visible log line.
    def generate_response(self, system_persona, user_instruction):
        messages = [
            {"role": "system", "content": system_persona},
            {"role": "user", "content": user_instruction}
        ]
        response = self.llm_client.chat.completions.create(
            model = config.LLM_MODEL_NAME,
            messages = messages,
            temperature = config.TEMP_TUTOR,          # was 0.1 in V1
            max_tokens = config.TUTOR_MAX_TOKENS       # V1 set no limit at all
        )
        final_answer = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            # V2.10: 4x was wrong -- confirmed live via a real 413 error that this Groq
            # account's actual ceiling is 6000 TPM total (prompt + completion combined), not
            # the much larger figure Groq's general docs page shows (that's for a paid tier
            # this account doesn't have). With keep_top=15, retrieved context alone is
            # already ~2,600-3,000 tokens, so 4x (8192) on top of that guaranteed a 413 and
            # crashed the run. +512 is a modest, safe bump that stays well under 6000 even
            # with the largest realistic context.
            retry_budget = config.TUTOR_MAX_TOKENS + 512
            print(f"  NOTE: response truncated at {config.TUTOR_MAX_TOKENS} tokens, "
                  f"retrying once with max_tokens={retry_budget}...")
            response = self.llm_client.chat.completions.create(
                model = config.LLM_MODEL_NAME,
                messages = messages,
                temperature = config.TEMP_TUTOR,
                max_tokens = retry_budget
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
