from deepeval.metrics import BiasMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate


def run_relevancy_evaluation(input_message: str, response_text: str, threshold=0.7):
    test_case = LLMTestCase(input=input_message, actual_output=response_text)
    metric = GEval(
        name="Relevancy With Clarifications",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        criteria=(
            "Judge if the response is relevant to the user's request. Mark as "
            "relevant when the response directly answers the request, provides a "
            "reasonable next step, or asks for missing required details. If the "
            "user provides a greeting (e.g., 'hi' or 'hello'), a friendly greeting "
            "or offer to help is relevant. If the user requests information about a "
            "flight and lacks essential "
            "flight details (airline, origin, destination, or time), a concise "
            "clarifying question is relevant. If the user asks a general delay "
            "question like 'will it be delayed', a delay prediction or explanation "
            "of delay likelihood is relevant even without specific flight details. "
            "Mark as not relevant only if the response ignores the request, "
            "changes topic, or provides completely unrelated information."
        ),
        model="gpt-3.5-turbo",
        threshold=threshold,
        strict_mode=True,
    )
    result = evaluate([test_case], [metric])

    md = result.test_results[0].metrics_data[0]
    return "success" if md.success else "fail"


def run_bias_evaluation(input_message: str, response_text: str, threshold=0.1):
    test_case = LLMTestCase(input=input_message, actual_output=response_text)
    metric = BiasMetric(
        model="gpt-3.5-turbo",
        threshold=threshold,
        strict_mode=True,
    )
    result = evaluate([test_case], [metric])

    md = result.test_results[0].metrics_data[0]
    return "success" if md.success else "fail"


def run_all_evaluations(input_message: str, response_text: str):
    relevancy_result = run_relevancy_evaluation(input_message, response_text)
    bias_result = run_bias_evaluation(input_message, response_text)
    return True if relevancy_result == "success" and bias_result == "success" else False
