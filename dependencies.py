import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))
