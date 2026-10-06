import os
from payload import payload
from crewai.tools import tool
from pathlib import Path
from dotenv import load_dotenv
from gen_ai_hub.proxy.native.sap.client import RPTClient
from crewai import Agent, Task, Crew

# Load .env from the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize RPT1 client after loading environment variables
rpt1_client = RPTClient()

@tool("call_rpt1")
def call_rpt1(payload: dict) -> str:
    """Call RPT-1 model to predict missing values in the payload.

    Args:
        payload: A dictionary containing the stolen items data with prediction placeholders.
                 This should be the exact payload provided in the task inputs.

    Returns:
        JSON string with predicted insurance values and item categories.
    """
    response = rpt1_client.predict(body=payload, model_name="sap-rpt-1-large")
    if response:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Create a Loss Appraiser Agent
appraiser_agent = Agent(
    role="Stolen Goods Loss Appraiser",
    goal="Assess the value of stolen items and provide a professional insurance appraisal report.",
    backstory="You are an experienced insurance appraiser specializing in fine art and valuables. You provide detailed assessments based on your expertise.",
    llm="sap/gpt-4o",  # provider/llm - Using one of the models from SAP's model library in Generative AI Hub
    tools=[call_rpt1],
    verbose=True
)

# Create a task for the appraiser
appraise_loss_task = Task(
    description="Provide a brief explanation of how an insurance appraiser would approach assessing stolen artwork and valuables.",
    expected_output="A professional explanation of the appraisal process.",
    agent=appraiser_agent
)

# Create a crew with the appraiser agent
crew = Crew(
    agents=[appraiser_agent],
    tasks=[appraise_loss_task],
    verbose=True
)

# Execute the crew
def main():
    result = crew.kickoff()
    print("\n" + "="*50)
    print("Insurance Appraiser Report:")
    print("="*50)
    print(result)

if __name__ == "__main__":
    main()


# Create a Loss Appraiser Agent
appraiser_agent = Agent(
    role="Loss Appraiser",
    goal=f"Predict the missing values of stolen items using the RPT-1 model via the call_rpt1 tool use this payload {payload} as input.",
    backstory="You are an expert insurance appraiser specializing in fine art valuation and theft assessment.",
    llm="sap/gpt-4o",  # provider/llm - Using one of the models from SAP's model library in Generative AI Hub
    verbose=True
)

# Create a task for the appraiser
appraise_loss_task = Task(
    description=f"Analyze the theft crime scene and predict the missing values of stolen items using the RPT-1 model via the call_rpt1 tool. Use this dict {payload} as input.",
    expected_output="JSON with predicted values for the stolen items.",
    agent=appraiser_agent
)
