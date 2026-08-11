'''
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

    def rewrite_query(self, user_query):
        prompt = f"Rewrite the following user query to use more precise academic terminology suitable for searching a University AI and Python syllabus. Output ONLY the rewritten query, nothing else.\nQuery: {user_query}"
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()


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


#==============================CODE BLOCK 2===================================================
import config
from app_init import llm_client

class CRAGEvaluator:
    def __init__(self, client):
        self.llm_client = client

    def evaluate_context(self, user_query, context_string):
        system_persona = """You are a strict retrieval evaluator for an Intelligent Tutoring System.
Your job is to measure Information Entailment. Choose ONE of three verdicts:

- CORRECT: The context contains the specific definitions, mechanics, code, or data needed to fully answer the User Query. A student with only this context could answer accurately.
- AMBIGUOUS: The context is on the right topic and contains partial information, but is incomplete -- key details, edge cases, or the full explanation are missing. A student with only this context would produce a shallow or partially correct answer.
- INCORRECT: The context is off-topic, is only structural metadata (headers/page numbers/table of contents), or does not contain the actual information required.

CRITICAL: Output ONLY one single word: CORRECT, AMBIGUOUS, or INCORRECT. No punctuation, no explanations."""

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
                temperature=0.0,
                max_tokens=10
            )
            evaluator_decision = response.choices[0].message.content.strip().upper()

            if "AMBIGUOUS" in evaluator_decision:
                return "AMBIGUOUS"
            elif "INCORRECT" in evaluator_decision:
                return "INCORRECT"
            elif "CORRECT" in evaluator_decision:
                return "CORRECT"
            else:
                return "INCORRECT"

        except Exception as e:
            print(f"Evaluator Error: {e}")
            return "INCORRECT"

    def rewrite_query(self, user_query):
        prompt = f"""The following user query failed to retrieve useful Python documentation on the first search attempt. Rewrite it to improve retrieval, following these rules:
- Preserve ALL code snippets, variable names, function names, and error message names EXACTLY as written.
- Preserve technical Python terms exactly (e.g., do not paraphrase 'f-string' as 'formatted string literal', do not change '__init__' to 'the init method').
- If the query is ambiguous, add clarifying Python-specific keywords, but do not change its intent.
- Do NOT invent citations, PEP numbers, section references, or facts not in the original query.
- Output ONLY the rewritten query itself. Do not prefix it with 'Query:', 'Rewritten:', or any other label.

Original query:
{user_query}"""
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    evaluator = CRAGEvaluator(llm_client)

    print("Testing the three-way interlock...")

    good_context = "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
    result_1 = evaluator.evaluate_context("What is machine learning?", good_context)
    print(f"Test 1 (good context): {result_1}")

    partial_context = "Machine learning is used in many applications including recommendation systems and image recognition. Popular ML libraries include scikit-learn and TensorFlow."
    result_2 = evaluator.evaluate_context("What is machine learning?", partial_context)
    print(f"Test 2 (partial context): {result_2}")

    bad_context = "S.NO Title 1 Introduction to Machine learning 2 Feature Selection"
    result_3 = evaluator.evaluate_context("What is machine learning?", bad_context)
    print(f"Test 3 (garbage context): {result_3}")

    code_query = "The code `fruits.index('banana', 3)` returns 3 but I expected the second banana at index 6. What's wrong?"
    rewritten = evaluator.rewrite_query(code_query)
    print(f"\nTest 4 (code-preserving rewrite):")
    print(f"  Original:  {code_query}")
    print(f"  Rewritten: {rewritten}")

#================================OUTPUT:
    .venv) admin@MacBookPro V1 % "/Users/admin/Data/UK/Ulster University/Curriculum/Semester 3/Dissertation/3. Technical Implementation/System A - LPITu
tor/My Implementation/V1/.venv/bin/python" "/Users/admin/Data/UK/Ulster University/Curriculum/Semester 3/Dissertation/3. Technical Implementation/Sys
tem A - LPITutor/My Implementation/V1/crag_evaluator.py"
/Users/admin/Data/UK/Ulster University/Curriculum/Semester 3/Dissertation/3. Technical Implementation/System A - LPITutor/My Implementation/V1/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
Testing the three-way interlock...
Test 1 (good context): CORRECT
Test 2 (partial context): CORRECT
Test 3 (garbage context): AMBIGUOUS

Test 4 (code-preserving rewrite):
  Original:  The code `fruits.index('banana', 3)` returns 3 but I expected the second banana at index 6. What's wrong?
  Rewritten: What's the behavior of the `index()` method in Python when searching for a value in a list starting from a specified index, and how does it handle duplicate values?
(.venv) admin@MacBookPro V1 % 
'''
# V2 work -- three-way CRAG evaluator (CORRECT / AMBIGUOUS / INCORRECT), replacing the
# binary version above. The binary version's evaluate_context and the older rewrite_query
# drafts are preserved untouched inside the docstring above (lines 1-187) for audit trail;
# this is the only active class in the file.
#
# V2.1 work -- evaluate_context() now runs on config.EVALUATOR_MODEL_NAME (openai/gpt-oss-20b),
# separate from the tutor's config.LLM_MODEL_NAME (llama-3.1-8b-instant). Originally the
# evaluator shared the tutor's model ("no separate judge model, per the bounded three-way
# scope decision") -- a disclosed limitation (REBUILD_SPEC: "V1 used the same 8B model for
# both, so failure modes were correlated by construction"). See config.py for the full
# rationale on the model choice. rewrite_query() below deliberately stays on
# config.LLM_MODEL_NAME -- the same-model-bias concern is specific to the evaluator marking
# its own homework (judging context it would itself generate from), not to query rewriting,
# so narrowing the swap to evaluate_context() only keeps this change minimal and testable.
import config
from app_init import llm_client

class CRAGEvaluator:
    def __init__(self, client):
        self.llm_client = client

    # v2 change: three-way verdict instead of binary. The binary version conflated two
    # different situations under INCORRECT -- "genuinely no relevant info" and "partial/
    # on-topic but incomplete" -- and V7's own results showed this caused over-rejection
    # (8 of 9 abstained items had System A context recall of 1.000). AMBIGUOUS gives the
    # partial-info case its own path instead of forcing a premature abstain.
    #
    # v2 recalibration (post-smoke-test): the first version of this prompt (copied from an
    # earlier draft already sitting in this file's history) scored a "talks around the topic
    # without defining it" context as CORRECT, and a bare table-of-contents line as AMBIGUOUS
    # -- confirmed 3/3 on a real smoke test. Both were one notch too lenient. Fixed by (1)
    # requiring CORRECT to directly answer the query, not just be topically detailed, and
    # (2) explicitly restoring the "only mentions the topic name" = INCORRECT language from
    # the original binary prompt, which had been dropped in the first 3-way draft.
    def evaluate_context(self, user_query, context_string):
        system_persona = """You are a strict retrieval evaluator for an Intelligent Tutoring System.
Your job is to judge whether the Context is sufficient to accurately and completely answer the User Query.
Choose ONE of three verdicts:

- CORRECT: The context DIRECTLY states the specific definition, mechanism, explanation, code, or data the query is asking for. A student with only this context could answer accurately and completely.
- AMBIGUOUS: The context is genuinely on-topic and contains real, substantive information related to the query, but does not directly state the core answer -- e.g. it discusses applications, examples, or related details while never giving the actual definition/explanation asked for.
- INCORRECT: The context does not contain the information asked for at all. This includes context that only mentions the topic by name without explaining it, bare headings, titles, table-of-contents lines, page numbers, or content that is off-topic entirely.

Before answering, check yourself: does the Context actually STATE the specific thing the Query asks for? If it only talks around the topic (uses, examples, related tools) without ever stating the core answer, that is AMBIGUOUS, not CORRECT. If it is just a title, heading, or topic name with no explanatory content, that is INCORRECT, not AMBIGUOUS.

Example: Query "What is a stack?"
- Context "Stacks are used in undo systems, expression evaluation, and function call management." -> AMBIGUOUS (real, relevant content, but never defines what a stack actually is).
- Context "3.2 Stacks and Queues" -> INCORRECT (just a heading, no explanation at all).
- Context "A stack is a linear data structure that follows Last-In-First-Out (LIFO) order, where elements are added and removed from the same end." -> CORRECT (directly states the definition).

CRITICAL: Output ONLY one single word: CORRECT, AMBIGUOUS, or INCORRECT. No punctuation, no explanations."""

        user_instruction = f"""
        User Query: {user_query}
        Context: {context_string}
        """
        try:
            # V2.1: openai/gpt-oss-20b is a REASONING model (unlike llama-3.1-8b-instant) --
            # by default it spends part of its token budget on an internal reasoning pass
            # before the final word. The old max_tokens=10 was tuned for a non-reasoning
            # model and would very likely get silently eaten by reasoning tokens here,
            # leaving .content empty -> falls through to the INCORRECT failsafe below ->
            # CRAG would silently reject every context. Fixed with reasoning_effort="low"
            # (this is a one-word classification, not a task that needs deep reasoning),
            # include_reasoning=False (keeps .content to just the verdict word, so the
            # parsing logic below doesn't need to change), and max_tokens raised to 200 as
            # headroom (Groq-specific params passed via extra_body since the official
            # openai SDK's typed create() doesn't recognize them as top-level kwargs).
            response = self.llm_client.chat.completions.create(
                model=config.EVALUATOR_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_persona},
                    {"role": "user", "content": user_instruction}
                ],
                temperature=0.0,  # zero creativity -- this is a classification call, not generation
                max_tokens=200,   # raised from 10 -- see reasoning-model note above
                extra_body={
                    "reasoning_effort": "low",
                    "include_reasoning": False,
                }
            )
            evaluator_decision = response.choices[0].message.content.strip().upper()

            # V2.1 defensive check: make the exact failure mode described above visible
            # instead of silently defaulting to INCORRECT with no trace.
            if not evaluator_decision:
                print("Evaluator WARNING: empty response content from "
                      f"{config.EVALUATOR_MODEL_NAME} -- likely reasoning tokens consumed "
                      "the token budget before the verdict word. Defaulting to INCORRECT.")

            # Order matters: check AMBIGUOUS and INCORRECT first, since a hallucinated
            # response containing multiple words could otherwise match "CORRECT" as a
            # substring of "INCORRECT". Failsafe default is INCORRECT (fail closed).
            if "AMBIGUOUS" in evaluator_decision:
                return "AMBIGUOUS"
            elif "INCORRECT" in evaluator_decision:
                return "INCORRECT"
            elif "CORRECT" in evaluator_decision:
                return "CORRECT"
            else:
                return "INCORRECT"

        except Exception as e:
            print(f"Evaluator Error: {e}")
            # If the API crashes, fail safely by tripping the interlock
            return "INCORRECT"

    def rewrite_query(self, user_query):
        prompt = f"""The following user query failed to retrieve useful Python documentation on the first search attempt. Rewrite it to improve retrieval, following these rules:
- Preserve ALL code snippets, exact variable names, exact numbers, function names, and error message names EXACTLY as written -- copy them verbatim into your rewrite, do not paraphrase or generalize them away.
- Preserve technical Python terms exactly (e.g., do not paraphrase 'f-string' as 'formatted string literal', do not change '__init__' to 'the init method').
- If the query is ambiguous, add clarifying Python-specific keywords, but do not change its intent or drop any specific detail (numbers, names, code) present in the original.
- Do NOT invent citations, PEP numbers, section references, or facts not in the original query.
- Output ONLY the rewritten query itself. Do not prefix it with 'Query:', 'Rewritten:', or any other label.

Original query:
{user_query}"""
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    evaluator = CRAGEvaluator(llm_client)

    print("Testing the three-way interlock...")

    good_context = "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
    result_1 = evaluator.evaluate_context("What is machine learning?", good_context)
    print(f"Test 1 (good context, expect CORRECT): {result_1}")

    partial_context = "Machine learning is used in many applications including recommendation systems and image recognition. Popular ML libraries include scikit-learn and TensorFlow."
    result_2 = evaluator.evaluate_context("What is machine learning?", partial_context)
    print(f"Test 2 (partial context, expect AMBIGUOUS): {result_2}")

    bad_context = "S.NO Title 1 Introduction to Machine learning 2 Feature Selection"
    result_3 = evaluator.evaluate_context("What is machine learning?", bad_context)
    print(f"Test 3 (garbage context, expect INCORRECT): {result_3}")

    code_query = "The code `fruits.index('banana', 3)` returns 3 but I expected the second banana at index 6. What's wrong?"
    rewritten = evaluator.rewrite_query(code_query)
    print(f"\nTest 4 (code-preserving rewrite):")
    print(f"  Original:  {code_query}")
    print(f"  Rewritten: {rewritten}")
    print(f"  Contains '3': {'3' in rewritten}, Contains '6': {'6' in rewritten}, Contains 'banana': {'banana' in rewritten}")