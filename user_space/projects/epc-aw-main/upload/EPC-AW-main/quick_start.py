# Import the solver
from MAS.epc_aw.solver import construct_solver
import os

# Set the LLM engine name
llm_engine_name = os.getenv("MODEL_Name") 

# Construct the solver
enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool", "Google_Search_Tool"]
tool_engine = [llm_engine_name] * len(enabled_tools)
solver = construct_solver(llm_engine_name=llm_engine_name, enabled_tools=enabled_tools, tool_engine=tool_engine, n=4, temperature=0.9, max_steps=4)

# Solve the user query
output = solver.solve("When is the director of film Les Tuche 2 's birthday?")
print(output["direct_output"])