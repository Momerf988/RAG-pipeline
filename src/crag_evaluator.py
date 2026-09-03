import config


class CRAGEvaluator:
    """The retrieval-quality interlock at the centre of System B. Judges whether retrieved
    context is actually sufficient to answer a question, and rewrites the query when it
    isn't, giving System B a second retrieval attempt before it ever generates an answer."""

    def __init__(self, client):
        self.llm_client = client

    # Three-way verdict rather than a binary pass/fail. A binary CORRECT/INCORRECT gate
    # conflates two different situations -- "genuinely no relevant information retrieved"
    # and "on-topic but incomplete" -- under the same INCORRECT label, and collapsing those
    # together causes over-rejection: context that partially covers a question gets thrown
    # away instead of used. AMBIGUOUS gives the partial-information case its own path
    # instead of forcing a premature abstain.
    #
    # The worked examples in the prompt below matter more than they look -- both verdict
    # boundaries (AMBIGUOUS vs CORRECT, and AMBIGUOUS vs INCORRECT) are genuinely easy for a
    # model to blur without them. Content that discusses a topic's applications without ever
    # stating its definition needs to land as AMBIGUOUS, not CORRECT or INCORRECT, and that
    # distinction only holds reliably once the prompt spells it out with a concrete example
    # rather than describing it abstractly.
    def evaluate_context(self, user_query, context_string):
        system_persona = """You are a strict retrieval evaluator for an Intelligent Tutoring System.
Your job is to judge whether the Context is sufficient to accurately and completely answer the User Query.
Choose ONE of three verdicts:

- CORRECT: The context DIRECTLY states the specific definition, mechanism, explanation, code, or data the query is asking for. A student with only this context could answer accurately and completely.
- AMBIGUOUS: The context is genuinely on-topic and contains real, substantive information related to the query, but does not directly state the core answer -- e.g. it discusses applications, examples, or related details while never giving the actual definition/explanation asked for.
- INCORRECT: The context does not contain the information asked for at all. This includes context that only mentions the topic by name without explaining it, bare headings, titles, table-of-contents lines, page numbers, or content that is off-topic entirely.

Before answering, check yourself: does the Context actually STATE the specific thing the Query asks for? If it only talks around the topic (uses, examples, related tools) without ever stating the core answer, that is AMBIGUOUS, not CORRECT. If it is just a title, heading, or topic name with no explanatory content, that is INCORRECT, not AMBIGUOUS.

Do NOT mark AMBIGUOUS content as INCORRECT just because it fails to state the core answer. INCORRECT is reserved for content with NO real substantive connection to the query at all -- bare headings, off-topic material, or content that only name-drops the topic. If the context names real, specific applications, tools, related concepts, or examples connected to the query's topic, that is substantive content and must be AMBIGUOUS, even though it does not answer the query directly.

Example: Query "What is a stack?"
- Context "Stacks are used in undo systems, expression evaluation, and function call management." -> AMBIGUOUS (real, relevant content, but never defines what a stack actually is).
- Context "3.2 Stacks and Queues" -> INCORRECT (just a heading, no explanation at all).
- Context "A stack is a linear data structure that follows Last-In-First-Out (LIFO) order, where elements are added and removed from the same end." -> CORRECT (directly states the definition).

Example: Query "What is machine learning?"
- Context "Machine learning is used in many applications including recommendation systems and image recognition. Popular ML libraries include scikit-learn and TensorFlow." -> AMBIGUOUS (names real, specific applications and tools -- genuinely substantive and on-topic -- but never states what machine learning actually is). This is NOT INCORRECT: it is more than a bare mention of the topic name.

CRITICAL: Output ONLY one single word: CORRECT, AMBIGUOUS, or INCORRECT. No punctuation, no explanations."""

        user_instruction = f"""
        User Query: {user_query}
        Context: {context_string}
        """
        try:
            # config.EVALUATOR_MODEL_NAME is a reasoning model, unlike the tutor -- it spends
            # part of its token budget on an internal reasoning pass before the final word,
            # so max_tokens needs real headroom (not the ~10 tokens a plain classifier call
            # would need) or the verdict word itself gets cut off. reasoning_effort="low" and
            # include_reasoning=False keep this a one-word classification call rather than a
            # deep reasoning task, and keep .content limited to just the verdict word so the
            # parsing below doesn't have to handle a reasoning trace mixed in.
            response = self.llm_client.chat.completions.create(
                model=config.EVALUATOR_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_persona},
                    {"role": "user", "content": user_instruction}
                ],
                temperature=0.0,
                max_tokens=200,
                extra_body={
                    "reasoning_effort": "low",
                    "include_reasoning": False,
                }
            )
            # Some backend providers return content: null (not an empty string) when a
            # reasoning model burns its whole token budget without emitting a verdict word.
            # `or ""` normalises both cases so the empty-response warning below always fires
            # instead of crashing straight into the except block with no trace of why.
            evaluator_decision = (response.choices[0].message.content or "").strip().upper()

            if not evaluator_decision:
                print("Evaluator WARNING: empty/None response content from "
                      f"{config.EVALUATOR_MODEL_NAME} -- likely reasoning tokens consumed "
                      "the token budget before the verdict word. Defaulting to INCORRECT.")

            # Check AMBIGUOUS and INCORRECT before CORRECT -- a hallucinated multi-word
            # response could otherwise match "CORRECT" as a literal substring of "INCORRECT".
            # Fails closed to INCORRECT by default, same as any parsing failure or API error.
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

    # Runs on the tutor's own model, not the evaluator model -- the same-model-bias concern
    # behind separating the evaluator from the generator is specifically about the evaluator
    # marking its own homework, which doesn't apply to query rewriting. Temperature 0 here
    # too, since a non-deterministic rewrite pulls a different second-pass retrieval and can
    # flip the second verdict on an otherwise identical question, which is exactly the kind
    # of run-to-run noise this project is trying to keep out of the comparison.
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
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
