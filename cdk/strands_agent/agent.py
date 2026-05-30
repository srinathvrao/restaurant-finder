from strands import Agent
import os
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands.tools.mcp import MCPClient
from bedrock_agentcore import BedrockAgentCoreApp
from strands.models import BedrockModel

# https://strandsagents.com/docs/user-guide/concepts/tools/mcp-tools/#aws-iam

MODEL_ID = "..."

SYSTEM_PROMPT = """
....
"""

GATEWAY_URL = os.environ["GATEWAY_URL"]

_restaurant_mcp_client = MCPClient(
    lambda: aws_iam_streamablehttp_client(
        endpoint=GATEWAY_URL,
        aws_region="us-east-1",
        aws_service="bedrock-agentcore",
    )
)
_restaurant_mcp_client.__enter__()
_restaurant_tools = _restaurant_mcp_client.list_tools_sync()

_model = BedrockModel(
    model_id=MODEL_ID,
    max_tokens=800,
)

app = BedrockAgentCoreApp()
@app.entrypoint
async def invoke(payload):
	""" AgentCore invocation for each request """

	user_prompt = payload.get("prompt", "")
	history = payload.get("history", [])
	location = payload.get("location")
	if location:
		user_prompt = f"[User's current location: lat={location['lat']}, lon={location['lon']}]\n\n{user_prompt}"

	# Convert stored history to Bedrock Converse message format for Strands
	prior_messages = [
		{"role": item["role"], "content": [{"text": item["content"]}]}
		for item in history
	]

	agent = Agent(
		model = _model,
		system_prompt=SYSTEM_PROMPT,
		tools = [*_restaurant_tools],
		messages = prior_messages,
		
	)

	# response streaming...
	async for event in agent.stream_async(user_prompt):
		if "data" in event:
			yield str(event["data"])
		elif "event" in event:
			yield event

if __name__ == "__main__":
	app.run()