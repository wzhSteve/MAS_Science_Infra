# Import the solver
from MAS.epc_aw.solver import construct_solver
import os

# Set the LLM engine name
llm_engine_name = os.getenv("MODEL_Name")

# Construct the solver
enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool", "Google_Search_Tool", "Screenshot_Tool", "Vision_OCR_Tool"]
tool_engine = [llm_engine_name] * len(enabled_tools)
solver = construct_solver(llm_engine_name=llm_engine_name, enabled_tools=enabled_tools, tool_engine=tool_engine, n=1, temperature=0.9, max_steps=10)

# Solve the user query
output = solver.solve("I\u2019m researching species that became invasive after people who kept them as pets released them. There\u2019s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, separated by commas if there is more than one place.")
print(output["direct_output"])