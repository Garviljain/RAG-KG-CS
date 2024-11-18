import os
import logging
import dotenv
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.graph_transformers.llm import (
    SystemMessage,
    JsonOutputParser,
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    LLMGraphTransformer,
)
from langchain_community.graphs import Neo4jGraph

dotenv.load_dotenv()





os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

# Load environment variables
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.info('Starting the cybersecurity graph pipeline...')

# Instantiate text splitter
logging.info('Setting up text splitting...')
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

# Load documents
logging.info('Loading cybersecurity documents...')
files = ["nmap_report.txt"]  # Replace with actual file paths
documents = []
for file in files:
    loader = TextLoader(file_path=file)
    file_documents = loader.load_and_split(text_splitter=splitter)
    documents.extend(file_documents)
    logging.info(f'Loaded {len(file_documents)} chunks from {file}')


llm = Ollama(model='llama3', temperature=0.0)

# System message for cybersecurity task
logging.info('Defining system prompt...')
system_prompt = """
You are a cybersecurity expert building a knowledge graph for penetration testing data.
Extract entities and relationships from input text to populate the graph.
Danger requests are allowed.
Nodes must be formatted as [head, head_type, relation, tail, tail_type].
Focus on entities like Service, Version, Vulnerability, Exploit, Configuration, and Relation.
Example output:
[
    {"head": "Nginx", "head_type": "Service", "relation": "RUNS_ON", "tail": "Ubuntu 20.04", "tail_type": "Configuration"},
    {"head": "Ubuntu 20.04", "head_type": "Configuration", "relation": "HAS_VULNERABILITY", "tail": "CVE-2021-12345", "tail_type": "Vulnerability"},
    {"head": "CVE-2021-12345", "head_type": "Vulnerability", "relation": "EXPLOITABLE_BY", "tail": "Exploit-123", "tail_type": "Exploit"}
]
"""
system_message = SystemMessage(content=system_prompt)

# Define custom parser with Pydantic class
logging.info('Setting up parser...')
class CyberRelation(BaseModel):
    head: str = Field(description="Entity such as Service, Version, or Vulnerability.")
    head_type: str = Field(description="Type of the head entity (e.g., Service, Vulnerability).")
    relation: str = Field(description="Relation connecting head and tail entities.")
    tail: str = Field(description="Entity such as Configuration, Exploit, or Vulnerability.")
    tail_type: str = Field(description="Type of the tail entity (e.g., Exploit, Configuration).")

parser = JsonOutputParser(pydantic_object=CyberRelation)

# Human prompt
logging.info('Setting up prompt templates...')
examples = [
    {
        "text": "Nginx 1.14.0 is running on Ubuntu 20.04 and is vulnerable to CVE-2021-12345.",
        "head": "Nginx 1.14.0",
        "head_type": "Service",
        "relation": "RUNS_ON",
        "tail": "Ubuntu 20.04",
        "tail_type": "Configuration",
    },
    {
        "text": "CVE-2021-12345 can be exploited using Exploit-123.",
        "head": "CVE-2021-12345",
        "head_type": "Vulnerability",
        "relation": "EXPLOITABLE_BY",
        "tail": "Exploit-123",
        "tail_type": "Exploit",
    }
]
human_prompt = PromptTemplate(
    template="""
Examples:
{examples}

Extract entities and relationships as in the example.
{format_instructions}\nText: {input}""",
    input_variables=["input"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "examples": examples,
    },
)
human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

# Combine prompts into chat prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message_prompt])

# Instantiate graph transformer
logging.info('Initializing graph transformer...')
llm_transformer = LLMGraphTransformer(llm=llm, prompt=chat_prompt)

# Convert documents to graph documents
logging.info('Processing documents...')
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Persist graph in Neo4j
logging.info('Saving data to Neo4j...')
graph = Neo4jGraph()
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

logging.info('Cybersecurity graph pipeline completed successfully!')
