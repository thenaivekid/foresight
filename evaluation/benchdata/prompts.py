"""Prompt templates copied verbatim from the benchmarks' own repositories.

Do NOT paraphrase these. The score is sensitive to the exact wording -- OVO's
CRR template in particular is the benchmark's phrasing of `have_enough_info`,
which is the thing we are measuring.

StreamingBench: src/benchmark/StreamingBench.py, StreamingBenchProactive.py
OVO-Bench:      constant.py
"""

# ---- StreamingBench --------------------------------------------------------
SB_MCQ = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options}'''
SB_MCQ_TAIL = "\n\nThe best option is:"

# The PO gating prompt LEAKS the ground-truth output on purpose -- that is the
# benchmark's design, and it is why PO scores timing only.
SB_PO_GATE = '''You are an advanced image question-answering AI assistant. You have been provided with images and a question related to the images. Your task is to carefully analyze the images and provide the answer to the question. You need to carefully confirm whether the images content meet the conditions of the question, and then output the correct content.

Question: {question} Is it the right time to output "{gt_output}"? You can only answer yes or no.

The answer is:
'''

# ---- OVO-Bench -------------------------------------------------------------
OVO_MCQ = '''
Question: {question}
Options:
{options}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response.
'''

OVO_REC = '''
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {question}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
'''

OVO_SSR = '''
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{step}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
'''

OVO_CRR = '''
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{question}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
'''


def fmt_options(options):
    """A./B./C./D. prefixes, added only if the data does not already carry them
    (StreamingBench's own driver does exactly this check)."""
    if options and str(options[0]).startswith("A."):
        return "\n".join(options)
    return "\n".join(f"{L}. {o}" for L, o in zip("ABCD", options))
