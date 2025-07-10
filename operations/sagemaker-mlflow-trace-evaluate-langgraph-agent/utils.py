from data import STOCK_PRICE
from graph import build_app

import dask.dataframe as dataframe
import multiprocessing.pool
import pandas as pd
import multiprocessing
from dotenv import load_dotenv
import os
from datetime import datetime

import mlflow
from mlflow.metrics import rouge1, rougeL, token_count, latency

from langchain_core.messages import HumanMessage


_GROUNDTRUTH_FILEMANE = f'file://{os.getcwd()}/golden_questions_answer.jsonl'
load_dotenv()
_MLFLOW_URI = os.getenv('MLFLOW_URI_SMAI')
_MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_ID')
_MLFLOW_LOGMODEL_ID = os.getenv('MLFLOW_LOGMODEL_ID')
_REGION = os.getenv('AWS_REGION')

mlflow.set_tracking_uri(_MLFLOW_URI)

# Enable system metrics logging (Optional)
# Ignore ROCm error
#mlflow.enable_system_metrics_logging()

# enable automatic logging of traces (Optional)
mlflow.langchain.autolog()

now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

@mlflow.trace(name="invokeAgent", attributes={"workflow": "utils_invokeAgent"}, span_type="utils_func")
def invokeAgent(query):
    print("\n Invoking Agent with prompt: ", query)
    messages = [HumanMessage(content=query)]
    react_graph_app = build_app()

    #Draw agent graph to local file
    react_graph_app.get_graph(xray=True).draw_mermaid_png(
        output_file_path="graph_diagram.png",  # Specify where to save the PNG image
        background_color="white", 
        padding=10
    )

    # Use agent
    result = react_graph_app.invoke({"messages": messages})
    return result["messages"]

def run_example_prompts():
    """
        Sample output:
    {'assistant': {'messages': [AIMessage(content=[{'type': 'text', 'text': "I'll help you find out the current price of Amazon (AMZN) stock. First, I'll retrieve the current stock price."}, {'type': 'tool_use', 'name': 'get_stock_price', 'input': {'symbol': 'AMZN'}, 'id': 'tooluse_m4RXhf8hRRG9rrRgGFM-pQ'}], additional_kwargs={}, response_metadata={'ResponseMetadata': {'RequestId': '879fce4b-7faf-48c4-b4e8-2463cc14fc31', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Thu, 20 Feb 2025 18:14:43 GMT', 'content-type': 'application/json', 'content-length': '404', 'connection': 'keep-alive', 'x-amzn-requestid': '879fce4b-7faf-48c4-b4e8-2463cc14fc31'}, 'RetryAttempts': 0}, 'stopReason': 'tool_use', 'metrics': {'latencyMs': [2953]}}, id='run-6459ef99-609e-42d3-b573-07b45dbc589b-0', tool_calls=[{'name': 'get_stock_price', 'args': {'symbol': 'AMZN'}, 'id': 'tooluse_m4RXhf8hRRG9rrRgGFM-pQ', 'type': 'tool_call'}], usage_metadata={'input_tokens': 433, 'output_tokens': 89, 'total_tokens': 522})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='{"price": 222.455}', name='get_stock_price', id='b2778333-8ad2-48f4-80fc-8518f11baccc', tool_call_id='tooluse_m4RXhf8hRRG9rrRgGFM-pQ')]}}
    ----
    {'assistant': {'messages': [AIMessage(content='The current price of AMZN stock is $222.455 per share. \n\nTo calculate the total price for 10 stocks:\n$222.455 × 10 = $2,224.55\n\nSo, 10 stocks of AMZN would currently cost $2,224.55. Please note that stock prices fluctuate constantly, so this price is just a snapshot of the current moment.\n\nIs there anything else you would like to know about AMZN stock?', additional_kwargs={}, response_metadata={'ResponseMetadata': {'RequestId': 'ebbd05be-5460-4646-9c1e-07e07a0465a0', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Thu, 20 Feb 2025 18:14:47 GMT', 'content-type': 'application/json', 'content-length': '547', 'connection': 'keep-alive', 'x-amzn-requestid': 'ebbd05be-5460-4646-9c1e-07e07a0465a0'}, 'RetryAttempts': 0}, 'stopReason': 'end_turn', 'metrics': {'latencyMs': [2481]}}, id='run-d3955356-c586-41d0-9bf6-7807493be183-0', usage_metadata={'input_tokens': 539, 'output_tokens': 114, 'total_tokens': 653})]}}
    ----

    """
    print(invokeAgent("What is the price of AMZN?"), "\n")
    print(invokeAgent("what is the capital of Canada?"), "\n")
    print(invokeAgent("What is the price of 10 stocks of AMZN?"), "\n")
    print(invokeAgent("What is the price of AMZN and AAPL?"), "\n")
    return

def agent_generate(inputs: pd.DataFrame, input_column) -> pd.DataFrame:
        prompt = inputs[input_column]
        full_response = invokeAgent(prompt)
        final_agent_answer = full_response[-1].content
        if final_agent_answer is None:
            raise KeyError("'content' key not present in message")
        inputs["answer"] = final_agent_answer
        return inputs

def _mlflow_groundtruth_data(inputs: pd.DataFrame) -> list[str]:
    return inputs["answer"].tolist()

def generate_evaluations(mlflow_uri, eval_filepath: str, agent_id: str = None, input_column: str = "inputs") -> pd.DataFrame:
    mlflow.set_tracking_uri(mlflow_uri)
    evaluation_dataset = pd.read_json(eval_filepath, lines=True)
    evaluation_dataset.reset_index(inplace=True)
    dataset = mlflow.data.from_pandas(
        evaluation_dataset, name="Langgraph agent evaluation input dataset"
    )
    mlflow.log_input(dataset, "prompt")
    print("Generating agent responses")
    parallel = dataframe.from_pandas(
        evaluation_dataset, npartitions=multiprocessing.cpu_count())
    agent_responses = parallel.apply(agent_generate, axis=1, meta=parallel,
                            input_column=input_column).compute() 
    print("Running mlflow evaluation")
    try:
        metrics = [latency(), rouge1(), rougeL(), token_count()]
        mlflow_eval_results = mlflow.evaluate(
            _mlflow_groundtruth_data,
            agent_responses,
            targets="ground_truth",
            model_type="question-answering",
            extra_metrics=metrics
        )
    except Exception as e:
        print(f"Error running mlflow evaluation: {e}")
        mlflow_eval_results = None
    return mlflow_eval_results, agent_responses

def get_experiment_id(experiment_name, create=True, tags=None):
    experiement = mlflow.get_experiment_by_name(experiment_name)
    if experiement is not None:
        experiment_id = experiement.experiment_id
    else:
        experiment_id = (mlflow.create_experiment(experiment_name, tags=tags)
                        if create else None)

    return experiment_id

def run_evaluation_mlflow(agent_file="graph.py", _REGISTER_AGENT_FLAG=True):
    with mlflow.start_run(
        experiment_id=get_experiment_id(_MLFLOW_EXPERIMENT_NAME), 
        run_name=timestamp, 
        tags={
            "project": os.getenv('PROJECT'),
            "model": os.getenv('MODELID'),
            "version": os.getenv('VERSION')
        }
    ):
        mlflow_eval_results, agent_inference_results = generate_evaluations(_MLFLOW_URI,
                                                                            _GROUNDTRUTH_FILEMANE)
        print(mlflow_eval_results.metrics.items())
        mlflow.log_artifact("graph_diagram.png")
        mlflow.log_param("domain", 'RegulatedFinance')
        mlflow.log_param("BU", 'Trading and Investment Banking')
        if _REGISTER_AGENT_FLAG:
            model_info = mlflow.langchain.log_model(
            lc_model=agent_file, # Path to our model Python file
            artifact_path=_MLFLOW_LOGMODEL_ID,
            )
            print("Agent registered successfully", model_info.model_uri)
    return mlflow_eval_results

def register_agent(agent_file):
    with mlflow.start_run(
        experiment_id=get_experiment_id(_MLFLOW_EXPERIMENT_NAME),
        run_name=timestamp,
        tags={
            "project": os.getenv('PROJECT'),
            "model": os.getenv('MODELID'),
            "version": os.getenv('VERSION')
        }
    ):
        model_info = mlflow.langchain.log_model(
        lc_model=agent_file, # Path to our model Python file
        artifact_path="langgraph",
    )

    model_uri = model_info.model_uri
    return model_uri