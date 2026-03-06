PROMETHEUS_CORRECTNESS_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a query, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is either 1 or 2 or 3 or 4 or 5. You should refer to the score rubric.
3. The output format should only look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.
5. Only evaluate on common things between generated answer and reference answer. Don't evaluate on things which are present in reference answer but not in generated answer.

###Instruction:
Your task is to evaluate the generated answer and reference answer for the following query:
{query}

###Generate answer to evaluate:
{generated_answer}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
Score 1: If the generated answer is not relevant to the user query and reference answer.
Score 2: If the generated answer is according to reference answer but not relevant to user query.
Score 3: If the generated answer is relevant to the user query and reference answer but contains mistakes.
Score 4: If the generated answer is relevant to the user query and has the exact same metrics as the reference answer, but it is not as concise.
Score 5: If the generated answer is relevant to the user query and fully correct according to the reference answer.

###Feedback:"""

PROMETHEUS_RELEVANCY_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a query with response, context, and a score rubric representing evaluation criteria are given.
1. You are provided with evaluation task with the help of a query with response and context.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is A or B. You should refer to the score rubric.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)”
5. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate: Your task is to evaluate if the response for the query is in line with the context information provided.

###Query and Response:
{query_str}

###Context:
{context_str}

###Score Rubrics:
Score YES: If the response for the query is in line with the context information provided.
Score NO: If the response for the query is not in line with the context information provided.

###Feedback: """

PROMETHEUS_FAITHFULNESS_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), an information, a context, and a score rubric representing evaluation criteria are given.
1. You are provided with evaluation task with the help of information, context information to give result based on score rubrics.
2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)”
5. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate: Your task is to evaluate if the given piece of information is supported by context.

###Information:
{query_str}

###Context:
{context_str}

###Score Rubrics:
Score YES: If the given piece of information is supported by context.
Score NO: If the given piece of information is not supported by context

###Feedback:"""